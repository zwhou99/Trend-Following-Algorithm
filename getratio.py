import numpy as np
import pandas as pd
from numba import njit


@njit
def compute_positions(HH, LL, bHH, sLL, sellShorts, buyLongs, High, Low, Open, Close, L, S, start, end, bars_back,
                      PV, slpg, E0):

    N = len(HH)  # Number of rows in the input data
    E = np.zeros(N) + E0  # Initialize equity array with initial equity value
    trades = np.zeros(N)  # Initialize trades array to record trade signals
    position = 0  # Initialize position (0: no position, 1: long, -1: short)

    # Iterate through the input data, starting from the maximum of bars_back + 1 and start
    for k in range(max(bars_back + 1, start), end):

        traded = False  # Initialize traded flag as False
        # Calculate the change in equity based on the current position and price change
        delta = PV * (Close[k] - Close[k - 1]) * position

        # If no position is held
        if position == 0:

            buy = buyLongs[k]  # Check if buy signal is present at the current index
            sell = sellShorts[k]  # Check if sell signal is present at the current index

            # If both buy and sell conditions are met
            if buy and sell:
                delta = -slpg + PV * (LL[k] - HH[k])  # Calculate change in equity
                trades[k] = 1  # Record trade signal
            else:
                # If buy condition is met
                if buy:
                    delta = -slpg / 2 + PV * (Close[k] - bHH[k])  # Calculate change in equity
                    position = 1  # Update position to long
                    traded = True  # Set traded flag to True
                    benchmarkLong = High[k]  # Set benchmark long price
                    trades[k] = 0.5  # Record trade signal
                # If sell condition is met
                elif sell:
                    delta = -slpg / 2 - PV * (Close[k] - sLL[k])  # Calculate change in equity
                    position = -1  # Update position to short
                    traded = True  # Set traded flag to True
                    benchmarkShort = Low[k]  # Set benchmark short price
                    trades[k] = 0.5  # Record trade signal

        # If a long position is held and not traded
        if position == 1 and not traded:
            sellShort = sellShorts[k]  # Check if sell short signal is present at the current index
            # Check if sell signal is present based on the benchmark long price and S value
            sell = Low[k] <= (benchmarkLong * (1 - S))

            # If sell short condition is met
            if sellShort:
                delta = delta - slpg - 2 * PV * (Close[k] - sLL[k])  # Calculate change in equity
                position = -1  # Update position to short
                benchmarkShort = Low[k]  # Set benchmark short price
                trades[k] = 1  # Record trade signal
            # If sell condition is met
            elif sell:
                delta = delta - slpg / 2 - PV * (
                        Close[k] - min(Open[k], (benchmarkLong * (1 - S))))  # Calculate change in equity
                position = 0  # Close position
                trades[k] = 0.5 # Record trade signal

            # Update benchmark long price with the maximum of the current High and the previous benchmark long price
            benchmarkLong = max(High[k], benchmarkLong)

        # If a short position is held and not traded
        if position == -1 and not traded:
            buyLong = buyLongs[k]  # Check if buy long signal is present at the current index
            # Check if buy signal is present based on the benchmark short price and S value
            buy = High[k] >= (benchmarkShort * (1 + S))

            # If buy long condition is met
            if buyLong:
                delta = delta - slpg + 2 * PV * (Close[k] - bHH[k])  # Calculate change in equity
                position = 1  # Update position to long
                benchmarkLong = High[k]  # Set benchmark long price
                trades[k] = 1  # Record trade signal
            # If buy condition is met
            elif buy:
                delta = delta - slpg / 2 + PV * (
                            Close[k] - max(Open[k], (benchmarkShort * (1 + S))))  # Calculate change in equity
                position = 0  # Close position
                trades[k] = 0.5  # Record trade signal

            # Update benchmark short price with the minimum of the current Low and the previous benchmark short price
            benchmarkShort = min(Low[k], benchmarkShort)

        # Update equity for the current index
        E[k] = E[k - 1] + delta

    return E, trades



def getratio(df, L, S, start, end, bars_back, PV, slpg, E0=100000):

    L = L * 10  # Scale L by multiplying by 10
    S = S / 1000  # Scale S by dividing by 1000

    # Calculate rolling maximum of High over the window defined by L
    df["HH"] = df.High.rolling(L, closed='left').max()
    # Calculate rolling minimum of Low over the window defined by L
    df["LL"] = df.Low.rolling(L, closed='left').min()
    # Calculate bHH as the maximum of HH and Open
    df['bHH'] = df[['HH', 'Open']].max(axis=1)
    # Calculate sLL as the minimum of LL and Open
    df['sLL'] = df[['LL', 'Open']].min(axis=1)

    # Identify sellShort signals when Low is less than or equal to LL
    df["sellShort"] = df.Low <= df.LL
    # Identify buyLong signals when High is greater than or equal to HH
    df["buyLong"] = df.High >= df.HH

    # Compute equity and trades using the compute_positions function with the input parameters
    E, trades = compute_positions(df.HH.values, df.LL.values, df.bHH.values, df.sLL.values, df.sellShort.values,
                                  df.buyLong.values, df.High.values, df.Low.values, df.Open.values, df.Close.values, L,
                                  S, start, end, bars_back, PV, slpg, E0)

    # Assign computed equity values to a new column in the DataFrame
    df["E"] = E
    # Calculate the running maximum of equity
    df["Emax"] = df.E.cummax()
    # Calculate drawdown by subtracting the running maximum equity from the equity
    df["DD"] = df.E - df.Emax

    # Calculate profit as the difference between equity at the end and equity at the start
    profit = df.E[end - 1] - df.E[start]
    # Calculate maximum drawdown as the absolute minimum of the drawdown array between start and end
    maxdrawdown = abs(min(df.DD[start: end]))

    # Calculate the performance ratio as profit divided by maximum drawdown
    # If profit or maxdrawdown is too small (less than 1e-3), set ratio to 0 to avoid division by a small number
    if abs(profit) < 1e-3 or maxdrawdown < 1e-3:
        ratio = 0
    else:
        ratio = profit / maxdrawdown

    # Return the performance ratio, profit, and maximum drawdown
    return ratio, profit, maxdrawdown

