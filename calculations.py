import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.express as px
import numpy as np
import math
from scipy.stats import skew, kurtosis, stats, norm

# Downloading Data------------------------------------------------------------------------------------------------------
stock = input("Escribí una accion: ").upper()
tickers = [f"{stock}","SPY"]
data = yf.download(tickers, start="2018-01-01")

# Variables---------------------------------------------------------------------------------------------------------------
asset = data["Close"][f"{stock}"]
benchmark = data["Close"]["SPY"]

asset_returns = asset.pct_change().dropna()
benchmark_returns = benchmark.pct_change().dropna()

n, minmax, mean, var, skew, kurt = stats.describe(asset_returns)
mini, maxi = minmax
std = var ** .5

# Daily return distribution---------------------------------------------------------------------------------------------------
def daily_returns_hist_plot():
    plt.hist(asset_returns, bins=25, edgecolor = "w", density=True, histtype='bar', color='blue', alpha=0.5)
    overlay = np.linspace(mini, maxi, 100)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.xlabel("Daily Returns")
    plt.ylabel("Frequency")
    plt.title("Histogram of Daily Returns")
    plt.plot(overlay, norm.pdf(overlay, mean, std))
    plt.xlim(-0.1,0.1)
    return plt.show()

daily_returns_hist_plot()

# Asset vs benchmark daily & yearly-------------------------------------------------------------------------------------------
def asset_daily_returns_vs_becnhmark():
    x = asset_returns.loc["2022-01-01":"2023-01-01"]
    y = benchmark_returns.loc["2022-01-01":"2023-01-01"]
    plt.plot(x, ls='-')
    plt.plot(y, ls='-')
    plt.ylabel("Daily Return")
    plt.title(f"Daily returns of {stock} vs benchmark SPY")
    plt.legend(tickers)
    plt.show()

asset_daily_returns_vs_becnhmark()

# Stock standard deviation----------------------------------------------------------------------------------------------------
def standard_deviation(prices, window=50, periods=252, clean=True):

    log_hl = (prices["High"] / prices["Low"]).apply(np.log)
    log_co = (prices["Close"] / prices["Open"]).apply(np.log)

    rs = 0.5 * log_hl ** 2 - (2 * math.log(2) - 1) * log_co ** 2

    def af(v):
        return (periods * v.mean()) ** 0.5

    result = rs.rolling(window=window, center=False).apply(func=af)
    if clean:
        return result.dropna()
    else:
        return result

sd = standard_deviation(data)
plt.plot(sd, ls='-')
plt.xlabel("Date")
plt.ylabel("Standard deviation")
plt.legend(tickers)
plt.show()

# Asset historical price with moving average---------------------------------------------------------------------------------
def asset_wma():
    a = asset.iloc[-500::]
    b = asset.rolling(20).mean().iloc[-500::]
    plt.plot(a, ls='-')
    plt.plot(b, ls='-')
    plt.xlabel("Date")
    plt.ylabel("Price")
    #fig = px.line(data, x = data.index, y = a, title = f"{stock} historical price")
    #fig.show()
    plt.title(f"Historical prices of {stock} with the 20 moving average")
    plt.show()
    return a, b

asset_wma()

# Sharpe ratio---------------------------------------------------------------------------------------------------------------
def sharpe_ratio(returns, adjustment_factor=0.0):
    returns_risk_adj = returns - adjustment_factor
    return (returns_risk_adj.mean() / returns_risk_adj.std()) * np.sqrt(252)

print(f"El Sharpe ratio es: {sharpe_ratio(asset_returns)}")
s = asset_returns.rolling(30).apply(sharpe_ratio)

plt.plot(s, ls='-')
plt.xlabel("Date")
plt.ylabel("Sharpe")
plt.title("Sharpe Ratio")
plt.show()


# Sortino ratio--------------------------------------------------------------------------------------------------------------
def sortino_ratio(returns, risk_free_return=0.0):
    
    # Numerator of the formula, annualized return
    returns_risk_adj = np.asanyarray(returns - risk_free_return)
    mean_annual_return = returns_risk_adj.mean() * 252

    # Denominator of the formula, downside deviation
    downside = np.clip(returns_risk_adj, a_min = np.NINF, a_max = 0)
    np.square(downside, out=downside)
    annualized_downside_deviation = np.sqrt(downside.mean()) * np.sqrt(252)
    
    return mean_annual_return / annualized_downside_deviation

print(f"El Sortino ratio es: {sortino_ratio(asset_returns)}")
sortino = asset_returns.rolling(30).apply(sortino_ratio)

plt.plot(sortino, ls='-')
plt.xlabel("Date")
plt.ylabel("Sortino")
plt.title("Sortino Ratio")
plt.show()

# Calculate skewness & kurtosis----------------------------------------------------------------------------------------------
#skewness = skew(asset_returns)
#kurt = kurtosis(asset_returns)

print(f"Skewness: {skew}")
print(f"Kurtosis: {kurt}")