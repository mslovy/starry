import tushare as ts
import numpy as np
import pandas as pd
import time

STOCKDIR = "stock_data"
HIST_DATA_PREFIX = "history_"

def normalize_stock_data(data):
    for i in range(1, data.shape[1]):
        column_data = data.iloc[:,i]
        mean = np.mean(column_data)
        std = np.std(column_data)
        data.iloc[:,i] = (column_data - mean) / std  # 标准化

class StockDataProcessor:

    def __init__(self, code, update = False):
        self._localtime = self._get_local_time()
        self._code = code
        self._file_name = STOCKDIR + "/" + HIST_DATA_PREFIX + self._code
        if (update):
            self._update_hist_data()

        self._stock_hist_data = self._load_hist_data()
        self._preprocess_hist_data()

    def _update_hist_data(self):
        data = ts.get_hist_data(self._code)
        data.to_csv(self._file_name)

    def _load_hist_data(self):
        return pd.read_csv(self._file_name)

    def _get_local_time(self):
        localtime = time.localtime(time.time())
        asctime = time.asctime(localtime).split(" ")
        return asctime

    def _preprocess_hist_data(self):
        self._stock_hist_data = self._stock_hist_data[::-1]  # 反转，使数据按照日期先后顺序排列'
        normalize_stock_data(self._stock_hist_data)
        self._stock_hist_data = self._stock_hist_data.drop('date',axis=1)

    def update_training_stock_data(self):
        pass
        #if(self._localtime.)

my = StockDataProcessor('000423', update = True)

print(my._stock_hist_data.head(5))