import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd
import streamlit as st


wind_direction = pd.read_csv('wind_direction.csv', index_col='datetime', parse_dates=['datetime']))

st.dataframe(wind_direction)
