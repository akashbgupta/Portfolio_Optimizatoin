import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import cufflinks as cf

sns.set_style("whitegrid")  # Set the seaborn style

pd.options.display.float_format = '{:.3f}'.format

listings = pd.read_csv("listings_clean.csv", index_col = "Symbol")

health = listings.loc[listings.Sector == "Health Care"].copy()
large_cap_hea = health.nlargest(n = 5, columns = "Market_Cap")
ticker = large_cap_hea.index.to_list()

#Create an appropriate Index from 1997 (start with 1996-12-31)
# until the end of 2023 that best reflects her strategy and
# create a normalized Price Chart (with Base Value 100 on 1996-12-31)!

start = "2000-01-01"
end = "2023-12-31"

stocks = yf.download(ticker, start, end)[['Adj Close', 'Close']]


#Finding the weights based on the Close price
weights = stocks.Close.div(stocks.Close.sum(axis = 1), axis = "index")


#Returns can be calculated using pct_change()
ret = stocks["Adj Close"].pct_change().dropna()

#calculation of index: (1 + (sum of weights*returns))) - cumulative product along time * 100

hea_index = ret.mul(weights.shift().dropna()).sum(axis = 1).add(1).cumprod().mul(100)
hea_index[pd.to_datetime("2000-12-31")] = 100 #Add 1990 Jan 1, since everything has been shifted down
hea_index.sort_index(inplace = True) # To bring Jan 1 1990 on top

hea_index.name = "Health Care"

#hea_index.plot(figsize = (12, 8))
#plt.show()

##Create a heatmap of all the 33 years

annual = hea_index.resample("A", kind = "period").last().to_frame() #Use the last close price from each year
annual.columns = ["Price"]
annual["Return"] = np.log(annual.Price / annual.Price.shift()) #log returns
annual.dropna(inplace = True)

years = annual.index.size

#Creating a 2D matrix of all the years
windows = [year for year in range(years, 0, -1)]
for year in windows:
    annual["{}Y".format(year)] = annual.Return.rolling(year).mean()

triangle = annual.drop(columns = ["Price", "Return"]) #Creating a triangle of returns across different time horizons
#Heatmap
#plt.figure(figsize=(25,20))
#sns.set(font_scale=0.4)
#sns.heatmap(triangle, annot = True, fmt = ".01%", cmap = "RdYlGn",
            #vmin = -0.10, vmax = 0.15, center = 0)
#plt.tick_params(axis = "y", labelright =True)
#plt.show()


## Comparing Healthcare with other sectors ##

indexes = pd.read_csv("sector_indexes.csv", parse_dates = ["Date"], index_col = "Date")
indexes["Health_Care"] = hea_index
indexes.Health_Care = indexes.Health_Care.div(indexes.Health_Care[0]).mul(100)

## Compare annualized risk and return based on daily (simple) returns and
# create an appropriate plot! Calculate the Sharpe Ratio and compare!

ret = indexes.pct_change().dropna()
#Function that returns annualised risks and returns with a gives returns database
def ann_risk_return(returns_df):
    summary = returns_df.agg(["mean", "std"]).T
    summary.columns = ["Return", "Risk"]
    summary.Return = summary.Return*252
    summary.Risk = summary.Risk * np.sqrt(252)
    return summary

summary = ann_risk_return(ret)

#Plot on risk return graph
#summary.plot(kind = "scatter", x = "Risk", y = "Return", figsize = (13,9), s = 50, fontsize = 15)
#for i in summary.index:
    #plt.annotate(i, xy=(summary.loc[i, "Risk"]+0.002, summary.loc[i, "Return"]+0.002), size = 15)
#plt.xlabel("ann. Risk(std)", fontsize = 15)
#plt.ylabel("ann. Return", fontsize = 15)
#plt.title("Risk/Return", fontsize = 20)
#plt.show()


rf = [0.013, 0] #Risk free return and risk
summary["Sharpe"] = (summary.Return - rf[0]) / summary.Risk
summary.sort_values("Sharpe", ascending = False)


# Could have improved the Sharpe Ratio of Portfolio by adding other Sectors, without increasing the Total Risk of the Portfolio!
# This is the Portfolio Diversification Effect!

noa = len(ret.columns)
nop = 50000
np.random.seed(111)
matrix = np.random.random(noa * nop).reshape(nop, noa) #creating a matrix of weights no. of industries*50,000
weights = matrix / matrix.sum(axis = 1, keepdims= True)

port_ret = ret.dot(weights.T) #Portfolio returns is a dot product of weights and returns
port_summary = ann_risk_return(port_ret)

#plt.figure(figsize = (15, 9))
#plt.scatter(port_summary.loc[:, "Risk"], port_summary.loc[:, "Return"],s= 20, color = "hotpink")
#plt.scatter(summary.loc[:, "Risk"], summary.loc[:, "Return"], s= 50, color = "black", marker = "D")
#plt.xlabel("ann. Risk(std)", fontsize = 15)
#plt.ylabel("ann. Return", fontsize = 15)
#plt.title("Risk/Return", fontsize = 20)
#plt.show()

port_summary["Sharpe"] = (port_summary.Return - rf[0]) / port_summary.Risk
port_summary.describe()

msrp = port_summary.Sharpe.idxmax() #Finding the optimal weights for max share ratio
msrp_w = weights[msrp, :]
pd.Series(index = indexes.columns, data = msrp_w)


## Identify Sectors with positive Alpha and a Beta-Factor < 1.
# Which Sectors will be added

SP500 = pd.read_csv("SP500_TR.csv", parse_dates = ["Date"], index_col = "Date")["Close"]
SP500 = SP500.reindex(indexes.index)
ret_SP = SP500.pct_change().dropna()
ret["SP500"] = ret_SP  #Add SP500 to the Return matrix

summary = ann_risk_return(ret)  #summary matrix now contains risk and return of SP500

summary["Sharpe"] = (summary["Return"].sub(rf[0]))/summary["Risk"]
summary["TotalRisk_var"] = np.power(summary.Risk, 2)
COV = ret.cov()*252  #Covariance matrix

summary["SystRisk_var"] = COV.iloc[:, -1]

summary["UnsystRisk_var"] = summary["TotalRisk_var"].sub(summary["SystRisk_var"])
summary["beta"] = summary.SystRisk_var / summary.loc["SP500", "SystRisk_var"]
summary["capm_ret"] = rf[0] + (summary.loc["SP500", "Return"] - rf[0]) * summary.beta
summary["alpha"] = summary.Return - summary.capm_ret


print(summary.loc[(summary.alpha > 0) & (summary.beta < 1)])

