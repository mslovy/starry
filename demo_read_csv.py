import pandas as pd

df = pd.read_csv("000423_hist_data_new.csv")
print(df)

df = pd.read_csv("000423_basics.csv", encoding="gbk")
print(df)