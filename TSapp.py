import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd
import streamlit as st
from datetime import datetime
import altair as alt
import re
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

    
energy=pd.read_csv('PJM_Load_hourly (1).csv', index_col=[0], parse_dates=[0])
E_d=energy["PJM_Load_MW"].asfreq('1d')#.shift(25)
dff=pd.DataFrame(E_d)
L=len(dff["PJM_Load_MW"])
c=5
a=0.1*np.array([10,.5,.5,.25,.1,.1])
b=0.1*np.array([10,.5,.5,.025,.01,-.1])
lam=np.array([2.1,.75,2.5,0.1*365/7,0.1*365/7,0.1*365/7])                   #[20,.5,.1,30*365/7,1*20*365/7,40*365/7])


t_=np.linspace(0,L/365,L)

f_t=np.array([fe(t_[i],c) for i in range(L)])

st.title("Consomation d'energie")
st.subheader("1.Visualisation")
st.markdown(f"{L}On commence par regarder la dinamique de notre série temporelle, on vous propose alors de choisir la fréquense à laquelle vous souhaitez moyenniser la serie. ")
# energy=pd.read_csv('PJM_Load_hourly (1).csv', index_col=[0], parse_dates=[0])
# E_d=energy["PJM_Load_MW"].asfreq('1d')#.shift(25)
# L=len(E_d["PJM_Load_MW"])
#st.dataframe(energy)
Freqs=['h','D','M']
option_freq='D'
option_freq=st.multiselect('veuillez selectionner une base pour la fréquence :',Freqs)
choix=re.sub(r"\s+", "", option_freq[0])
#num=st.slider( f"For the parameter: {k}",step= (l[1]-l[0]),min_value=l[0], max_value=l[-1],value= l[-1]) 

    #fr=energy["PJM_Load_MW"].asfreq(f"{1}{option_freq}") 
fr=pd.DataFrame(energy["PJM_Load_MW"].asfreq(f"{1}{choix}"))
frr=fr.copy()
frr['ind']=[k for k in range(len(fr["PJM_Load_MW"]) )]
frr.reset_index(inplace=True)

base1=alt.Chart(pd.DataFrame(frr))
line=base1.mark_line().encode(
    x='Datetime',
    y='PJM_Load_MW'
    )
st.altair_chart(line, use_container_width=True)
#st.dataframe(frr)
st.subheader("1. Modélisation")
st.markdown("Dans cette partie nous alons essayer de modeliser la série temporelle, on propose alors d'écrire la consomation d'enérgie $C_t$ sous la forme: ")
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

st.markdown("Etant donnée des observations de prix $(S_{t_0},...,S_{t_n})$ On peut ignorer en premier lieux l'effect de retour a la moyenne pour éstimer les paramétres de $f(t)=ln(S_t)$ qui minimise")

st.latex(r'''
    \sum_{i=0}^n(ln(S_{t_i})-f(t_i))^2
     ''')
st.markdown("Maintenant étant donéé $f$, on calibre le processus $(X_t)_{t\geq0}$ par la méthode du maximum de vraisemblanc en utilisant $\left\{ln(S_{t_0})-f(t_0),....,ln(S_{t_n})-f(t_n)\right\}$") 

#f_t=np.array([fe(t_[i],c) for i in range(733 )])
ou_=orstein_uhlenbeck2(1,L ,0,100,0,10)
#energy["PJM_Load_MW"].asfreq('1d').plot()
s_t=23000*np.random.uniform(0.95,1,L)+10*np.exp(f_t+0.5*ou_)
#E_d=energy["PJM_Load_MW"].asfreq('1d')#.shift(25)
#dff=pd.DataFrame(E_d)
ind=dff.index
df_es=pd.DataFrame(s_t)
#df_es.columns=['Datetime','O_U']
df_es.columns=['estim']
df_es.set_index(ind)
dff["estim"]=s_t

test=dff.copy()
test['ind']=[k for k in range(L)]
test.reset_index(inplace=True)
line1=alt.Chart(test).mark_line().encode(
    x='Datetime',
    y='PJM_Load_MW'
)
line2=alt.Chart(test).mark_line(color='red').encode(
    x='Datetime',
    y='estim'
)
st.altair_chart(line1+line2, use_container_width=True)
#wind_direction = pd.read_csv('wind_direction.csv', index_col='datetime', parse_dates=['datetime'])

#st.dataframe(wind_direction)
