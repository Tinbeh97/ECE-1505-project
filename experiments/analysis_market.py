# analyze on fetched data
# please use fetch_market_data.py to fetch data
#
# argument: <symbol file path> (use 1d interval data)

from fetch_market_data import *

if __name__ == "__main__":

    assert(len(sys.argv)>1)
    symbol_file_path = sys.argv[1]
    print("using symbol file: " + symbol_file_path)

    data = None

    with open(symbol_file_path) as data_file:
        data = json.load(data_file)

    assert(data is not None)

    #fetch
    (data_name, tickers, time_window, interval) = (data["name"],
                                                   data["tickers"],
                                                   data["date_window"],
                                                   data["interval"])
    print("data name: ", data_name)
    print("tickers: ", tickers)
    print("time_window: ", str(time_window))
    print("interval: ", interval)

    record_data = 'fetched/'+data_name+'_'+time_window[0]+'_'+time_window[1]+'.npy'
    record_tickers = 'fetched/'+data_name+'_'+time_window[0]+'_'+time_window[1]+'_tickers.npy'

    #check saved data
    arr = np.load(record_data)
    tk_names = np.loadtxt(record_tickers, dtype='str')

    # print(arr)
    print(tk_names)
    print("data dim: ", arr.shape)

    num_data_points = arr.shape[1]
    print("num_data_points: ", num_data_points)
    analysis_interval = 7
    indexing_monthly = np.arange(0,num_data_points,analysis_interval)
    
    closing = arr[:,indexing_monthly,3]    
    opening = arr[:,indexing_monthly,0]
    
    delta_price_frac = (closing-opening)/opening

    #expect 1d interval 
    index_day = interval.find("1d")
    assert(index_day != -1)
    x_label = "week"

    plot_data(tk_names, delta_price_frac, x_label, "delta price fraction", is_log_scale=True)

    print(delta_price_frac.shape)

    
