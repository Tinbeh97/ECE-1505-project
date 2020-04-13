# produces files contining tickers in json format
# saves files in symbols/ folder

import json

tickers_sp100_2016 = {
    "name": "sp100_2016",
    "date_window": ["2014-01-01", "2018-12-31"],
    "tickers":
    ['AAPL','ABBV','ABT','ACN','AGN','AIG','ALL','AMGN','AMZN','AXP','BA','BAC','BIIB','BK','BLK','BMY','C','CAT','CELG','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DHR','DIS','DOW','DUK','EMC','EMR','EXC','F','FB','FDX','FOX','FOXA','GD','GE','GILD','GM','GOOG','GOOGL','GS','HAL','HD','HON','IBM','INTC','JNJ','JPM','KMI','KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON','MRK','MS','MSFT','NEE','NKE','ORCL','OXY','PCLN','PEP','PFE','PG','PM','PYPL','QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP','UPS','USB','USD','UTX','V','VZ','WBA','WFC']
}

tickers_sp100_2020_april_6 = {
    "name": "sp100_2020_april_6",
    "date_window": ["2014-01-01", "2018-12-31"],
    "tickers":
    ['AAPL','ABBV','ABT','ACN','ADBE','AGN','AIG','ALL','AMGN','AMT','AMZN','AXP','BA','BAC','BIIB','BK','BKNG','BLK','BMY','BRK-B','C','CAT','CHTR','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DHR','DIS','DUK','EMR','EXC','F','FB','FDX','GD','GE','GILD','GM','GOOG','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM','KMI','KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MRK','MS','MSFT','NEE','NFLX','NKE','NVDA','ORCL','OXY','PEP','PFE','PG','PM','PYPL','QCOM','RTX','SBUX','SLB','SO','SPG','T','TGT','TMO','TXN','UNH','UNP','UPS','USB','V','VZ','WBA','WFC','WMT','XOM']}

l = [tickers_sp100_2016, tickers_sp100_2020_april_6]

for i in l:
    
    with open('symbols/'+i["name"]+'.json', 'w', encoding='utf-8') as f:
        json.dump(i, f, ensure_ascii=False, indent=4)
    
    with open('symbols/'+i["name"]+'.json') as data_file:
        data_loaded = json.load(data_file)
        assert(i == data_loaded)
        print("saved: " + str(data_loaded))
