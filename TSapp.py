import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd
import streamlit as st
from datetime import datetime
#import statsmodels.api as sm
# from numpy.random import normal, seed
# from scipy.stats import norm
# from statsmodels.tsa.arima_model import ARMA
# from statsmodels.tsa.stattools import adfuller

# from statsmodels.tsa.arima_process import ArmaProcess
# from statsmodels.tsa.arima_model import ARIMA
import math
#from sklearn.metrics import mean_squared_error
from math import *
from numpy.random import *



st.title("Consomation d'energie")
st.subheader("1.Visualisation")
st.markdown("On commence par regarder la dinamique de notre série temporelle, on vous propose alors de choisir la fréquense à laquelle vous souhaitez moyenniser la serie. ")
energy=pd.read_csv('PJM_Load_hourly (1).csv', index_col=[0], parse_dates=[0])
st.dataframe(energy)
Freqs=['h','D','M']
option_freq='none'
option_freq=st.multiselect('veuillez selectionner une base pour la fréquence :',Freqs)

#num=st.slider( f"For the parameter: {k}",step= (l[1]-l[0]),min_value=l[0], max_value=l[-1],value= l[-1]) 
if (option_freq!='none'):
    #fr=energy["PJM_Load_MW"].asfreq(f"{1}{option_freq}") 
    fr=pd.DataFrame(energy["PJM_Load_MW"].asfreq(f"{1}{option_freq}"))
    frr=fr.copy()
    frr['ind']=[k for k in range(len(frr["PJM_Load_MW"]) )]
    frr.reset_index(inplace=True)

    base1=alt.Chart(frr)
    line1=base1.mark_line().encode(
        x='Datetime',
        y='PJM_Load_MW'
        )
    st.altair_chart(line1, use_container_width=True)

st.subheader("1. Modélisation")
st.markdown("Dans cette partie nous alons utiliser le processus de retour à la moyenne d'*Ornstein Uhlenbeck* qui suit la dynamique suivante:")
st.latex(r'''
     dX_t=\theta(\mu-X_t)dt+\sigma dB_t
     ''')

#wind_direction = pd.read_csv('wind_direction.csv', index_col='datetime', parse_dates=['datetime'])

#st.dataframe(wind_direction)
