import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd


google = pd.read_csv('../input/stock-time-series-20050101-to-20171231/GOOGL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])

st.dataframe(google)
