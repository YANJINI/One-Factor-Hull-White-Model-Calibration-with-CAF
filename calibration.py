import pandas as pd
import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
from scipy.optimize import minimize, fmin
from scipy.interpolate import CubicSpline
from functools import partial


class OneFactorHW():

    def __init__(self, discount_df, capflr_df, alpha=0.001, sigma=0.0001):
        # nested class
        self.calc_tools = self.HWCalculationTools(self)
        self.rate_tools = self.RatesTools(self)
        self.visualize = self.Visualize(self)

        # two main data
        self.initial_yc = self.load_initial_yield_curve(discount_df)
        self.capflrs = self.load_capflr(capflr_df)

        # initial values for parameters
        self.alpha = alpha
        self.sigma = sigma

        # optimized values for parameters
        self.opt_alpha = None
        self.opt_sigma = None
        self.opt_res = None

        # default interpolation method
        self.intpol = "cubic"

        # simulation default options
        self.max_maturity = 30
        self.dt = 1 / 12
        self.n_paths = 1000

    def load_initial_yield_curve(self, discount_df):
        """ Creates a dictionary containing discount factors and zero rates of initial yield curve.
        Args:
            discount_df: a dataframe of discount factors
        """
        df = np.array(discount_df['Discount'])
        t = np.array(discount_df['year fraction'])
        zerorates = self.rate_tools.df_to_zerorate(df, t)

        return namedtuple('initial_yc', ['df', 't', 'zerorates'])(df, t, zerorates)

    def load_capflr(self, capflr_df):
        """ Create lists of cap and floor namedtuple
        Args:
            capflr_df: a dataframe of cap and floor data
        """
        capflr = []
        for idx, row in capflr_df.iterrows():
            start_t = np.arange(row['freq'], row['maturity'], row['freq'])
            end_t = start_t + row['freq']

            capflr.append(namedtuple(
                row['capflr_id'],
                ['freq', 'maturity', 'market_price', 'norm_vol', 'strike',
                 'notional', 'capflr_type', 'start_t', 'end_t'])
                          (row['freq'], row['maturity'], row['market_price'],
                           row['norm_vol'], row['strike'] / 100, row['notional'],
                           row['capflr_id'][:3], start_t, end_t))

        return capflr

    def split_train_test(self):
        """Split cap and floor data into two sets, one for calibration (cap data), the other for test (floor data)"""
        # cap data for calibration
        train_data = [capflr for capflr in self.capflrs if capflr.capflr_type == 'cap']
        # test data for test
        test_data = [capflr for capflr in self.capflrs if capflr.capflr_type == 'flr']

        return train_data, test_data

    def calibrate(self):
        """ Calibrates Hull-White model to given instruments and initial term structure"""
        data_calib, _ = self.split_train_test()

        t0 = 0.
        r0 = self.rate_tools.zerorate_t(t0)

        def opt_func(x, t0, r0):
            self.alpha, self.sigma = x

            sse = 0
            for capflr in data_calib:
                model_price = self.calc_tools.get_model_analytical_price(t0, r0, capflr)
                err = (model_price - capflr.market_price) ** 2
                sse += err

            return np.sqrt(sse / len(self.capflrs))

        res = minimize(opt_func, (self.alpha, self.sigma), (t0, r0), method='Nelder-Mead',
                       bounds=[(0, None), (0, None)])

        self.opt_alpha, self.opt_sigma, self.opt_res = (res.x[0], res.x[1], res)

    class RatesTools:

        def __init__(self, hw_obj):
            self.hw = hw_obj

        def df_t(self, t, intpol=None):
            """ Interpolates discount factors given t using given a interpolation method
            Args:
                t: t for which discount factors are interpolated
                intpol: interpolation method to be used
            """
            if intpol is None:
                intpol = self.hw.intpol

            if intpol == "linear":
                return np.interp(t, self.hw.initial_yc.t, self.hw.initial_yc.df)
            elif intpol == "log_linear":
                return np.exp(np.interp(t, self.hw.initial_yc.t, np.log(self.hw.initial_yc.df)))
            elif intpol == "cubic":
                df = CubicSpline(self.hw.initial_yc.t, self.hw.initial_yc.df, bc_type="natural")(t)
                if df.ndim == 0:
                    df = float(df)
                return df

        def zerorate_t(self, t):
            """ Calculates zero rates given t
            Args:
                t: t for which discount factors are interpolated
            """
            df_t = self.df_t(t)
            zerorate = self.df_to_zerorate(df_t, t)

            return zerorate

        def df_to_zerorate(self, df, t):
            """ Convert discount factors to zero rates given t using continuouly-compounding.
            Args:
                df: an array of discount factors
                t: year fraction between valuation date and each maturity date
            """
            if isinstance(df, np.ndarray) and isinstance(t, float):
                zerorates = -np.log(df) / t
                return zerorates

            if isinstance(df, np.ndarray):
                copy_df = df.copy()
                copy_t = t.copy()
                if len(df.shape) == 1:
                    if df[0] == 1 and t[0] == 0:
                        copy_df[0] = df[1]
                        copy_t[0] = t[1]
                else:
                    if all(copy_df[:, 0] == 1) and copy_t[0] == 0:
                        copy_df[:, 0] = copy_df[:, 1]
                        copy_t[0] = copy_t[1]

                zerorates = -np.log(copy_df) / copy_t

            elif isinstance(df, float):
                if df == 1 and t == 0:
                    df = self.hw.initial_yc.df[1]
                    t = self.hw.initial_yc.t[1]
                zerorates = -np.log(df) / t

            return zerorates

        def f_0_t(self, t, dt=1e-6, intpol=None):
            """ Calculates market instantaneous forward rate, f(0, t) = f(0, t, t+dt) when dt goes to zero
            Args:
                t: year fraction of time when forward rate contract starts
                dt: discretized time step
                intpol = interpolation method to be used
            """
            if intpol is None:
                intpol = self.hw.intpol

            df_t = self.df_t(t, intpol)
            df_t_dt = self.df_t(t + dt, intpol)
            return self.df_to_zerorate(df_t_dt / df_t, dt)

        def f_0_t_dt(self, t, dt=1e-6, intpol=None):
            """ Calculates the partial derivative of market instantaneous forward rate, f(0, t) / dt
            Args:
                t: year fraction of time when forward rate contract starts
                dt: discretized time step
                intpol = interpolation method to be used
            """
            if intpol is None:
                intpol = self.hw.intpol

            f_0_t1 = self.f_0_t(t - dt, dt=dt, intpol=intpol)
            f_0_t2 = self.f_0_t(t + dt, dt=dt, intpol=intpol)

            return (f_0_t2 - f_0_t1) / 2 * dt

        def simulate_short_rates(self):
            """ Generates simulation of short rates from calibrated hull white model using Euler Maruyama discretization scheme
            """
            a = self.hw.opt_alpha
            s = self.hw.opt_sigma
            num_steps = int(self.hw.max_maturity / self.hw.dt)

            t = np.arange(num_steps + 1) * self.hw.dt
            short_rates = np.zeros((self.hw.n_paths, num_steps + 1))
            short_rates[:, 0] = self.zerorate_t(0.)

            np.random.seed(1)
            for i in range(1, num_steps + 1):
                # 0 <= s < t
                time_s = t[i - 1]
                time_t = t[i]
                alpha_s = self.hw.calc_tools.g_t(time_s)
                alpha_t = self.hw.calc_tools.g_t(time_t)
                r_s = short_rates[:, i - 1]

                Z = np.random.normal(0, 1, self.hw.n_paths)
                cond_mean = r_s * np.exp(-a * (time_t - time_s)) + alpha_t - alpha_s * np.exp(-a * (time_t - time_s))
                cond_sig = np.sqrt(s ** 2 / 2 / a * (1 - np.exp(-2 * a * (time_t - time_s))))
                short_rates[:, i] = cond_mean + cond_sig * Z

            return t, short_rates

        def short_rates_to_df(self, short_rates, t):
            """ Convert simulated short rates to forward discount factors (prices of ZCB)
            Args:
                short_rates: simulated short rates
                t: time steps in a path
            """
            df = np.ones_like(short_rates)
            fwd_df = self.hw.calc_tools.P(r=short_rates[:, :-1], t=t[:-1], T=t[1:])
            df[:, 1:] = np.cumprod(fwd_df, axis=1)

            return df

    class HWCalculationTools:

        def __init__(self, hw_obj):
            self.hw = hw_obj

        def get_model_analytical_price(self, t, r, capflr):
            """ Calculates the price of a Cap or Floor at valuation time t
            Args:
                t: valuation time
                r: instantaneous short rate at t
                capflr: a dictionary with Cap or Floor info
            """
            dt = capflr.end_t - capflr.start_t
            K = 1 + capflr.strike * dt

            captlets_floorlets = capflr.notional * K * self.ZCBO(t, T=capflr.start_t, S=capflr.end_t,
                                                                 X=1 / K, r=r, type=capflr.capflr_type)

            return np.sum(captlets_floorlets)

        def ZCBO(self, t, T, S, X, r, type):
            """ Calculates the price of an European call/put option on Zero Coupon Bond
            Args:
                t: year fraction of valudation time
                T: year fraction of option maturity time
                S: year fraction of ZCB maturity time
                X: strike price
                r: instantaneous short rate at time t
                type: option type (call or put)
                """
            h = self.h(t, T, S, X, r)
            sig_p = self.sigma_p(t, T, S)
            sign = 1 if type == 'call' else -1

            return sign * (self.P(r, t, S) * norm.cdf(sign * h) - X * self.P(r, t, T) * norm.cdf(sign * (h - sig_p)))

        def h(self, t, T, S, X, r):
            """ Helper function in HW ZCBO formula
            Args:
                t: year fraction of valudation time
                T: year fraction of option maturity time
                S: year fraction of ZCB maturity time
                X: strike price
                r: instantaneous short rate at time t
            """
            sig_p = self.sigma_p(t, T, S)
            P_t_T = self.P(r, t, T)
            P_t_S = self.P(r, t, S)

            return 1 / sig_p * np.log(P_t_S / P_t_T / X) + sig_p / 2

        def sigma_p(self, t, T, S):
            """ Helper function in HW ZCBO formula
            Args:
                t: year fraction of valudation time
                T: year fraction of option maturity time
                S: year fraction of ZCB maturity time
                """
            s = self.hw.sigma
            a = self.hw.alpha
            B_T_S = self.B(T, S)

            return s * np.sqrt((1 - np.exp(-2 * a * (T - t))) / 2 / a) * B_T_S

        def P(self, r, t, T):
            """ Calculates zero coupon bond price using Hull-White affine term structure
            Args:
                r: instantaneous short rate at t
                t: year fraction of valuation time
                T: year fraction of ZCB maturity
            """
            return self.A(t, T) * np.exp(-self.B(t, T) * r)

        def A(self, t, T):
            """ Calculates A(t, T) in Hull-White affine term structure zero coupon bond price P(t, T) = A(t, T) * e^{-B(t, T) * r_t}
            Args:
                t: year fraction of valuation time
                T: year fraction of ZCB maturity
            """
            df_t = self.hw.rate_tools.df_t(t)
            df_T = self.hw.rate_tools.df_t(T)
            fM = self.hw.rate_tools.f_0_t(t)
            B_t_T = self.B(t, T)
            a = self.hw.alpha
            s = self.hw.sigma

            return df_T / df_t * np.exp(B_t_T * fM - s ** 2 / 4 / a * (1 - np.exp(-2 * a * t)) * B_t_T ** 2)

        def B(self, t, T):
            """ Calculates B(t, T) in Hull-White affine term structure zero coupon bond price P(t, T) = A(t, T) * e^{-B(t, T) * r_t}
            Args:
                t: year fraction of valuation time
                T: year fraction of ZCB maturity
            """
            a = self.hw.alpha
            return (1 / a) * (1 - np.exp(-a * (T - t)))

        def g_t(self, t):
            """ Helper function g(t) in conditional disctribution of r(t)"""
            f = self.hw.rate_tools.f_0_t(t)
            a = self.hw.opt_alpha
            s = self.hw.opt_sigma
            return f + 0.5 * (s / a * (1 - np.exp(-a * t))) ** 2

    class Visualize:

        def __init__(self, hw_obj):
            self.hw = hw_obj

        def plot_initial_yc(self, plot_df=False):
            if plot_df:
                x1 = self.hw.initial_yc.df
                y_label = "Discount Factor"
            else:
                x1 = self.hw.initial_yc.zerorates * 100
                y_label = "Zero rates (%)"

            plt.figure(figsize=(10, 6))
            plt.plot(self.hw.initial_yc.t, x1, marker='o', linestyle='-', markersize=4)
            plt.title("ESTR Yield Curve on 1st April 2024", fontsize=16)
            plt.xlabel('Tenor', fontsize=14)
            plt.ylabel(y_label, fontsize=14)
            plt.grid(True)
            plt.legend()
            plt.show()

        def plot_capflr(self):
            train, test = self.hw.split_train_test()
            cap_maturities = [cap.maturity for cap in train]
            floor_maturities = [floor.maturity for floor in test]
            cap_nvol = [cap.norm_vol / 100 for cap in train]
            floor_nvol = [floor.norm_vol / 100 for floor in test]

            plt.figure(figsize=(10, 6))
            plt.plot(cap_maturities, cap_nvol, label="Cap implied normal volatility", marker='o', linestyle='-',
                     markersize=4)
            plt.plot(floor_maturities, floor_nvol, label="Floor implied normal volatility", marker='o', linestyle='-',
                     markersize=4)
            plt.title("At-The-Money ESTR Interest Rate Cap and Floor Implied Normal Volatility")

            plt.xlabel('Tenor', fontsize=14)
            plt.ylabel('Implied Normal Volatility', fontsize=14)
            plt.grid(True)
            plt.legend()
            plt.show()

        def plot_m_fwd_rate(self, dt=1e-6, intpol=None):
            if intpol is None:
                intpol = self.hw.intpol

            plt.figure(figsize=(10, 6))
            plt.plot(self.hw.initial_yc.t[1:], self.hw.initial_yc.zerorates[1:] * 100,
                     label="Initial term structure (Zero rates)", marker='o', linestyle='-', markersize=4)
            plt.plot(np.arange(361) * 1 / 12, self.hw.rate_tools.f_0_t(np.arange(361) * 1 / 12, dt, intpol) * 100,
                     label="Market instantaneous forward rate", linestyle='-', markersize=4)
            plt.title("Market Instantaneous Forward Rate and Initial Term Structure")
            plt.xlabel('Tenor', fontsize=14)
            plt.ylabel('Interest Rate (%)', fontsize=14)
            plt.grid(True)
            plt.legend()
            plt.show()

        def plot_zr_comparison(self, keep_focus=True):
            t, mc_short_rates = self.hw.rate_tools.simulate_short_rates()
            mc_df = self.hw.rate_tools.short_rates_to_df(mc_short_rates, t)
            mc_zero_rates = self.hw.rate_tools.df_to_zerorate(mc_df, t)
            mc_zero_rates[:, 0] = mc_zero_rates[:, 1]

            mean_mc_zero_rates = np.mean(mc_zero_rates, axis=0)

            plt.figure(figsize=(10, 6))
            plt.plot(self.hw.initial_yc.t, self.hw.initial_yc.zerorates * 100, label='Initial Term Structure',
                     marker='o')
            plt.plot(t, mean_mc_zero_rates * 100, label='Simulated Mean Zero Rates (Hull-White)', linestyle='-',
                     color='orange')
            if keep_focus:
                plt.ylim([min(min(self.hw.initial_yc.zerorates * 100), min(mean_mc_zero_rates * 100)) * 0.95,
                          max(max(self.hw.initial_yc.zerorates * 100), max(mean_mc_zero_rates * 100)) * 1.05])

            for confidence_level, color in zip([0.95, 0.9, 0.8, 0.5], ['orange', 'red', 'pink', 'yellow', 'green']):
                lower_bound = np.percentile(mc_zero_rates, (1 - confidence_level) / 2 * 100, axis=0)
                upper_bound = np.percentile(mc_zero_rates, (1 + confidence_level) / 2 * 100, axis=0)
                plt.fill_between(t, lower_bound * 100, upper_bound * 100, color=color, alpha=1,
                                 label=f'{confidence_level * 100}% Confidence Interval')
            plt.xlabel('Maturity (Years)')
            plt.ylabel('Interest Rate (%)')
            plt.title(
                f'Comparison of Initial Term Structure and Simulated Zero Rates (alpha: {self.hw.opt_alpha:.4f}, sigma: {self.hw.opt_sigma:.4f})')
            plt.legend()
            plt.grid(True)
            plt.show()

        def plot_short_rates_zero_rates(self):
            t, mc_short_rates = self.hw.rate_tools.simulate_short_rates()
            mean_mc_short_rates = np.mean(mc_short_rates, axis=0)
            mc_df = self.hw.rate_tools.short_rates_to_df(mc_short_rates, t)
            mc_zero_rates = self.hw.rate_tools.df_to_zerorate(mc_df, t)
            mc_zero_rates[:, 0] = mc_zero_rates[:, 1]
            mean_mc_zero_rates = np.mean(mc_zero_rates, axis=0)

            plt.figure(figsize=(10, 6))
            # Plot the lines with smaller markers and pastel colors
            plt.plot(self.hw.initial_yc.t, self.hw.initial_yc.zerorates * 100,
                     label='Initial Term Structure (Zero rates)', marker='o', linestyle='-', markersize=4)
            plt.plot(t, mean_mc_short_rates * 100,
                     label='Simulated Mean Short Rates (Hull-White)', linestyle='-')
            plt.plot(t, mean_mc_zero_rates * 100,
                     label='Simulated Mean Zero Rates (Hull-White)', linestyle='-')

            plt.xlabel('Tenor', fontsize=14)
            plt.ylabel('Interest Rate (%)', fontsize=14)
            plt.title(
                f'Simulated Short Rates and Zero Rates (alpha: {self.hw.opt_alpha:.4f}, sigma: {self.hw.opt_sigma:.4f})')
            plt.legend()
            plt.grid(True)
            plt.show()

        def plot_short_rates_mfwd_rates(self, dt=1e-6, intpol=None):
            if intpol is None:
                intpol = self.hw.intpol

            t, mc_short_rates = self.hw.rate_tools.simulate_short_rates()
            mean_mc_short_rates = np.mean(mc_short_rates, axis=0)

            plt.figure(figsize=(10, 6))
            plt.plot(self.hw.initial_yc.t, self.hw.initial_yc.zerorates * 100,
                     label='Initial Term Structure (Zero rates)', marker='o', linestyle='-', markersize=4)
            plt.plot(t, mean_mc_short_rates * 100,
                     label='Simulated Mean Short Rates (Hull-White)', linestyle='-')
            plt.plot(np.arange(361) * 1 / 12, self.hw.rate_tools.f_0_t(np.arange(361) * 1 / 12, dt, intpol) * 100,
                     label="Market instantaneous forward rate", linestyle='-', markersize=4)

            plt.xlabel('Tenor', fontsize=14)
            plt.ylabel('Interest Rate (%)', fontsize=14)
            plt.title(
                f'Simulated Short Rates and Market Instantaneous Forward Rates (alpha: {self.hw.opt_alpha:.4f}, sigma: {self.hw.opt_sigma:.4f})')
            plt.legend()
            plt.grid(True)
            plt.show()

        def plot_price_comparison(self, is_train=True):
            train, test = self.hw.split_train_test()
            if is_train:
                capflrs = train
            else:
                capflrs = test
            t0 = 0.
            r0 = self.hw.rate_tools.zerorate_t(t0)
            model_prices = []
            market_prices = []
            maturities = []
            for capflr in capflrs:
                model_prices.append(self.hw.calc_tools.get_model_analytical_price(t0, r0, capflr))
                market_prices.append(capflr.market_price)
                maturities.append(capflr.maturity)

            plt.figure(figsize=(10, 6))
            plt.plot(maturities, model_prices, label='Model Prices', marker='o')
            plt.plot(maturities, market_prices, label='Market Prices', linestyle='--', marker='o', color='orange')

            for i, maturity in enumerate(maturities):
                difference = model_prices[i] - market_prices[i]
                plt.annotate(f'{difference:,.0f}', (maturity, model_prices[i]), textcoords="offset points",
                             xytext=(0, 10), ha='center')

            formatter = FuncFormatter(lambda x, pos: f'{x:,.0f}')
            plt.gca().yaxis.set_major_formatter(formatter)
            plt.xlabel('Tenor', fontsize=14)
            plt.ylabel('Prices (EUR)', fontsize=14)
            plt.title(
                f'Comparison of Model Prices and Market Prices for {"Train (Cap)" if is_train else "Test (Floor)"} Data',
                fontsize=16)
            plt.legend()
            plt.grid(True)
            plt.show()


if __name__ == '__main__':
    discount_df = pd.read_csv('./data/ESTR_df.csv')
    capflr_df = pd.read_csv('./data/ESTR_capflr.csv')
    initial_a, initial_sig = 0.05, 0.0001
    hw = OneFactorHW(discount_df, capflr_df, initial_a, initial_sig)

    hw.calibrate()
    hw.visualize.plot_initial_yc(plot_df=False)
    hw.visualize.plot_initial_yc(plot_df=True)
    hw.visualize.plot_m_fwd_rate(intpol="linear")
    hw.visualize.plot_m_fwd_rate(intpol="log_linear")
    hw.visualize.plot_m_fwd_rate(intpol="cubic")
    hw.visualize.plot_capflr()
    hw.visualize.plot_price_comparison(is_train=True)
    hw.visualize.plot_price_comparison(is_train=False)
    hw.visualize.plot_short_rates_zero_rates()
    hw.visualize.plot_short_rates_mfwd_rates()
