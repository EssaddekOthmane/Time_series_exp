import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd
import streamlit as st
from datetime import datetime
import statsmodels.api as sm
from numpy.random import normal, seed
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARIMA
import math
from sklearn.metrics import mean_squared_error
from math import *
from numpy.random import *




# energy=pd.read_csv('PJM_Load_hourly.csv(1)', index_col=[0], parse_dates=[0])

# Freqs=['h','D','M']
# option=st.multiselect('Select the three models you want to compaire :',Freqs)
# num=st.slider( f"For the parameter: {k}",step= (l[1]-l[0]),min_value=l[0], max_value=l[-1],value= l[-1]) 

# fr=energy["PJM_Load_MW"].asfreq('1d').plot() 



#wind_direction = pd.read_csv('wind_direction.csv', index_col='datetime', parse_dates=['datetime'])

st.dataframe(wind_direction)
