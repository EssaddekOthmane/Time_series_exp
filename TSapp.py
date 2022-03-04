import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd


wind_direction = pd.read_csv('wind_direction.csv')

st.dataframe(wind_direction)
