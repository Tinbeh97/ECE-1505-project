# produces files contining tickers in json format
# saves files in symbols/ folder

import json

#select some stocks, using s&p100 at april 6,2020, see their history from 2014 to 2018
tickers_sp100_2014_2018_1d = {
    "name": "sp100_2014_2018_1d",
    "date_window": ["2014-01-01", "2018-12-31"],
    "interval": "1d",
    "tickers":
    ['AAPL','ABBV','ABT','ACN','ADBE','AGN','AIG','ALL','AMGN','AMT','AMZN','AXP','BA','BAC','BIIB','BK','BKNG','BLK','BMY','BRK-B','C','CAT','CHTR','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DHR','DIS','DUK','EMR','EXC','F','FB','FDX','GD','GE','GILD','GM','GOOG','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM','KMI','KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MRK','MS','MSFT','NEE','NFLX','NKE','NVDA','ORCL','OXY','PEP','PFE','PG','PM','PYPL','QCOM','RTX','SBUX','SLB','SO','SPG','T','TGT','TMO','TXN','UNH','UNP','UPS','USB','V','VZ','WBA','WFC','WMT','XOM']}

#see only 1 day history at 1min intervals
tickers_sp100_2020_03_25_1m = {
    "name": "sp100_2020_03_25_1m",
    "date_window": ["2020-03-25", "2020-03-26"],
    "interval": "1m",
    "tickers":
    ['AAPL','ABBV','ABT','ACN','ADBE','AGN','AIG','ALL','AMGN','AMT','AMZN','AXP','BA','BAC','BIIB','BK','BKNG','BLK','BMY','BRK-B','C','CAT','CHTR','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DHR','DIS','DUK','EMR','EXC','F','FB','FDX','GD','GE','GILD','GM','GOOG','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM','KMI','KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MRK','MS','MSFT','NEE','NFLX','NKE','NVDA','ORCL','OXY','PEP','PFE','PG','PM','PYPL','QCOM','RTX','SBUX','SLB','SO','SPG','T','TGT','TMO','TXN','UNH','UNP','UPS','USB','V','VZ','WBA','WFC','WMT','XOM']}

#see only 1 day history at 2min intervals
tickers_sp100_2020_03_25_2m = {
    "name": "sp100_2020_03_25_2m",
    "date_window": ["2020-03-25", "2020-03-26"],
    "interval": "2m",
    "tickers":
    ['AAPL','ABBV','ABT','ACN','ADBE','AGN','AIG','ALL','AMGN','AMT','AMZN','AXP','BA','BAC','BIIB','BK','BKNG','BLK','BMY','BRK-B','C','CAT','CHTR','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DHR','DIS','DUK','EMR','EXC','F','FB','FDX','GD','GE','GILD','GM','GOOG','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM','KMI','KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MRK','MS','MSFT','NEE','NFLX','NKE','NVDA','ORCL','OXY','PEP','PFE','PG','PM','PYPL','QCOM','RTX','SBUX','SLB','SO','SPG','T','TGT','TMO','TXN','UNH','UNP','UPS','USB','V','VZ','WBA','WFC','WMT','XOM']}

tickers_small_2014_2018_1d = {
    "name": "small_2014_2018_1d",
    "date_window": ["2014-01-01", "2018-12-31"],
    "interval": "1d",
    "tickers":
    ['GOOGL','AMZN','REGN','AMGN','TM','HMC','EA','ATVI']}

#so far, using this one in experiment for monthly, weekly, biweekly
tickers_small_2000_2018_1d = {
    "name": "small_2000_2018_1d",
    "date_window": ["2000-01-01", "2018-12-31"],
    "interval": "1d",
    "tickers":
    ['GOOGL','AMZN','REGN','AMGN','TM','HMC','EA','ATVI']}

#database query seems to return NaN for monthly, so use daily instead
# tickers_small_2014_2018_1mo = {
#     "name": "small_2014_2018_1mo",
#     "date_window": ["2014-01-01", "2018-12-31"],
#     "interval": "1mo",
#     "tickers":
#     ['GOOGL','AMZN','REGN','AMGN','GM','F','EA','ATVI']}

tickers_tiny_2014_2018_1d = {
    "name": "tiny_2014_2018_1d",
    "date_window": ["2014-01-01", "2018-01-01"],
    "interval": "1d",
    "tickers":
    ['GOOGL','AMZN','REGN','AMGN']}

l = [tickers_sp100_2014_2018_1d,
     tickers_sp100_2020_03_25_1m,
     tickers_sp100_2020_03_25_2m,
     tickers_small_2014_2018_1d,
     tickers_small_2000_2018_1d,
     tickers_tiny_2014_2018_1d,
     # tickers_small_2014_2018_1mo
]

for i in l:
    
    with open('symbols/'+i["name"]+'.json', 'w', encoding='utf-8') as f:
        json.dump(i, f, ensure_ascii=False, indent=4)
    
    with open('symbols/'+i["name"]+'.json') as data_file:
        data_loaded = json.load(data_file)
        assert(i == data_loaded)
        print("saved: " + str(data_loaded))
