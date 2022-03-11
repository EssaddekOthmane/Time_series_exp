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



def orstein_uhlenbeck2(T,n,X_0,theta,u,sigma):
    db=np.random.normal(0,sqrt(T/n),n)
    dt=T/n
    X=np.zeros(n)
    X[0]=X_0
    for i in range(1,n):
        X[i]=X[i-1]+ theta*(u-X[i-1])*dt+sigma*db[i-1]
    return(X)


def fe(x,c):
    f=c
    for i in range(5):
        f=f+a[i]*(1+np.cos(2*np.pi*lam[i]*x))+b[i]*(np.sin(0.3333*2*np.pi*lam[i]*x))
    return(f)

c=5
a=0.1*np.array([10,1,.5,.1,.01,.001])
b=0.1*np.array([10,1,.5,.1,.01,.001])
lam=np.array([1.8,.005,.25,366/7,366/7,366/7])                   #[20,.5,.1,30*365/7,1*20*365/7,40*365/7])


t_=np.linspace(0,1,366 )

f_t=np.array([fe(t_[i],c) for i in range(366 )])

st.title("Consomation d'energie")
st.subheader("1.Visualisation")
st.markdown("On commence par regarder la dinamique de notre série temporelle, on vous propose alors de choisir la fréquense à laquelle vous souhaitez moyenniser la serie. ")
energy=pd.read_csv('PJM_Load_hourly (1).csv', index_col=[0], parse_dates=[0])
#st.dataframe(energy)
Freqs=['h','D','M']
option_freq='D'
#option_freq=st.multiselect('veuillez selectionner une base pour la fréquence :',Freqs)

#num=st.slider( f"For the parameter: {k}",step= (l[1]-l[0]),min_value=l[0], max_value=l[-1],value= l[-1]) 

    #fr=energy["PJM_Load_MW"].asfreq(f"{1}{option_freq}") 
fr=pd.DataFrame(energy["PJM_Load_MW"].asfreq(f"{1}{option_freq}"))
frr=fr.copy()
frr['ind']=[k for k in range(len(fr["PJM_Load_MW"]) )]
frr.reset_index(inplace=True)

# base1=alt.Chart(pd.DataFrame(frr))
# line1=base1.mark_line().encode(
#     x='Datetime',
#     y='PJM_Load_MW'
#     )
# st.altair_chart(line1, use_container_width=True)
st.dataframe(frr)
st.subheader("1. Modélisation")
st.markdown("Dans cette partie nous alons essayyer de modeliser la série temporelle, on propose alors d'écrire la consomation d'enérgie $C_t$ sous la forme: ")
st.latex(r'''
    C_t=exp\{f(t)+X_t\}
     ''')
st.markdown("Pour modéliser les saisonabilité, on utilise $f$:")
st.latex(r'''
    f(t)=c+\sum_{i=1}^6a_icos(2\pi\lambda_it)
     ''')

st.markdown("On rajoute aussi le processus de retour à la moyenne d'*Ornstein Uhlenbeck* qui suit la dynamique suivante:")
st.latex(r'''
     dX_t=\theta(\mu-X_t)dt+\sigma dB_t
     ''')


ou_=orstein_uhlenbeck2(1,366 ,0,100,0,8)
#energy["PJM_Load_MW"].asfreq('1d').plot()
s_t=25000*np.random.uniform(0.95,1,366)+5*np.exp(f_t+0.5*ou_)#f_t*(1+0.005*(np.exp(ou_)+.001*f_t*ou_))
E_d=energy["PJM_Load_MW"]['1999-01-01':'2000-01-01'].asfreq('1d')#.shift(25)
dff=pd.DataFrame(E_d)
ind=E_d.index
df_es=pd.DataFrame(s_t)
#df_es.columns=['Datetime','O_U']
df_es.columns=['estim']
df_es.set_index(ind)
dff["estim"]=s_t

test=dff.copy()
test['ind']=[k for k in range(366 )]
test.reset_index(inplace=True)
line1=alt.Chart(test).mark_line().encode(
    x='Datetime',
    y='PJM_Load_MW'
)
line2=alt.Chart(test).mark_line(color='red').encode(
    x='Datetime',
    y='estim_sh'
)
st.altair_chart(line1+line2, use_container_width=True)
#wind_direction = pd.read_csv('wind_direction.csv', index_col='datetime', parse_dates=['datetime'])

#st.dataframe(wind_direction)
