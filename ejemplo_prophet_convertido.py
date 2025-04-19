#!/usr/bin/env python
# coding: utf-8

# In[2]:


from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from prophet.diagnostics import cross_validation, performance_metrics


# In[3]:


def create_model(cp,cps=0.05,sps=10.0,sm='additive',g='linear',iw=0.9):
  model = Prophet(
    growth=g,
    changepoints=cp,
    changepoint_prior_scale=cps,
    seasonality_prior_scale=sps,
    seasonality_mode=sm,
    interval_width=iw
  )
  return model

def calc_err(fc,dataframe):
  df_fc = fc[["ds", "yhat"]].merge(dataframe, on="ds")
  df_fc = df_fc.dropna(subset=["y", "yhat"])

  mae = mean_absolute_error(df_fc["y"], df_fc["yhat"])
  mse = mean_squared_error(df_fc["y"], df_fc["yhat"])
  rmse = np.sqrt(mse)

  print(f"MAE: {mae:.2f}")
  print(f"MSE: {mse:.2f}")
  print(f"RMSE: {rmse:.2f}")

  smape = (np.abs(df_fc["y"] - df_fc["yhat"]) / ((np.abs(df_fc['y']) + np.abs(df_fc['yhat']))/2)).mean() * 100
  print(f"SMAPE: {smape:.2f}%")

def print_fc(fc,n):
  print(fc[['ds','yhat','yhat_lower','yhat_upper']].tail(n))


# In[4]:


sheet_id = '1xTnusy_z4rhm3-M_A4tFjDDiKlg8ZHdCU_NxC3p0Wak'
sheet_name = 'Entregas'
url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'

og_df = pd.read_csv(url)


# In[48]:


og_df.head(5)


# In[98]:


df = og_df.loc[:,['Date','Demanda Total']]
df_black = og_df.loc[:,['Date','Black']];df_black=df_black.iloc[6:,:]; df_black['floor']=0; df_black['cap']=df_black['Black'].max()*1.5
df_op = og_df.loc[:,['Date','Ocean Plastic']]; df_op=df_op.iloc[13:,:]; df_op['floor']=0;df_op['cap']=df_op['Ocean Plastic'].max()*1.5
df_cb = og_df.loc[:,['Date','Cardjolote Black']]; df_cb=df_cb.iloc[10:,:]; df_cb['floor']=0;df_cb['cap']=df_cb['Cardjolote Black'].max()*1.5


# In[99]:


df['Date']  = pd.to_datetime(df['Date'])
df_black['Date'] = pd.to_datetime(df['Date'])
df_op['Date'] = pd.to_datetime(df['Date'])
df_cb['Date'] = pd.to_datetime(df['Date'])

df = df.rename(columns={'Date':'ds','Demanda Total':'y'})
df_black = df_black.rename(columns={'Date':'ds','Black':'y'})
df_op = df_op.rename(columns={'Date':'ds','Ocean Plastic':'y'})
df_cb = df_cb.rename(columns={'Date':'ds','Cardjolote Black':'y'})


# In[100]:


changepoints = [
    '2023-05-01',
    '2023-07-01',
    '2023-10-01',
    '2024-05-01',
    '2024-09-01',
    '2024-11-01'
]


# In[101]:


model       = create_model(changepoints,0.21,7,'additive')
model_black = create_model(changepoints[1:],0.9,15,sm='additive',g='logistic')
model_op    = create_model(df_op['ds'],2,30,sm='multiplicative',g='logistic')
model_cb    = create_model(df_cb['ds'],10,100,'multiplicative',g='logistic')


# In[102]:


import logging

# Para que no se vean los warnings
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

model.fit(df)
model_black.fit(df_black)
model_op.fit(df_op)
model_cb.fit(df_cb)


# In[103]:


n_periods_to_predict = 6

future = model.make_future_dataframe(periods=n_periods_to_predict,freq='MS')
future_black = model_black.make_future_dataframe(periods=n_periods_to_predict,freq='MS');future_black['floor']=0;future_black['cap']=df_black['cap'].iloc[0]
future_op = model_op.make_future_dataframe(periods=n_periods_to_predict,freq='MS');future_op['floor']=0;future_op['cap']=df_op['cap'].iloc[0]
future_cb = model_cb.make_future_dataframe(periods=n_periods_to_predict,freq='MS');future_cb['floor']=0;future_cb['cap']=df_cb['cap'].iloc[0]

forecast = model.predict(future)
forecast_black = model_black.predict(future_black)
forecast_op = model_op.predict(future_op)
forecast_cb = model_cb.predict(future_cb)


# In[104]:


fig = model.plot(forecast)
for cp in model.changepoints:
    plt.axvline(cp,color='r',ls='--',alpha=0.5)
plt.plot(forecast[-n_periods_to_predict-1:]['ds'],forecast[-n_periods_to_predict-1:]['yhat'],c='red')
plt.title("Predicción de Demanda Total (con Prophet)")
plt.xlabel("Fecha")
plt.ylabel("Demanda Total")
plt.tight_layout()
plt.show()


# In[45]:


fig = model_black.plot(forecast_black)
for cp in model_black.changepoints:
    plt.axvline(cp,color='r',ls='--',alpha=0.5)
plt.plot(forecast[-n_periods_to_predict-1:]['ds'],forecast_black[-n_periods_to_predict-1:]['yhat'],c='red')
plt.title("Predicción de Demanda Black (con Prophet)")
plt.xlabel("Fecha")
plt.ylabel("Demanda Black")
plt.tight_layout()
plt.show()


# In[46]:


fig = model_op.plot(forecast_op)
for cp in model_op.changepoints:
    plt.axvline(cp,color='r',ls='--',alpha=0.5)
plt.plot(forecast[-n_periods_to_predict-1:]['ds'],forecast_op[-n_periods_to_predict-1:]['yhat'],c='red')
plt.title("Predicción de Demanda Ocean Plastic (con Prophet)")
plt.xlabel("Fecha")
plt.ylabel("Demanda Ocean Plastic")
plt.tight_layout()
plt.show()


# In[47]:


fig = model_cb.plot(forecast_cb)
for cp in model_cb.changepoints:
    plt.axvline(cp,color='r',ls='--',alpha=0.5)
plt.plot(forecast[-n_periods_to_predict-1:]['ds'],forecast_cb[-n_periods_to_predict-1:]['yhat'],c='red')
plt.title("Predicción de Demanda Cardjolote Black (con Prophet)")
plt.xlabel("Fecha")
plt.ylabel("Demanda Cardjolote Black")
plt.tight_layout()
plt.show()


# In[105]:


print_fc(forecast,n_periods_to_predict)


# In[17]:


print_fc(forecast_black,n_periods_to_predict)


# In[18]:


print_fc(forecast_op,n_periods_to_predict)


# In[19]:


print_fc(forecast_cb,n_periods_to_predict)


# In[106]:


print('Errores de demanda total')
calc_err(forecast,df)
print('\nErrores de Black')
calc_err(forecast_black,df_black)
print('\nErrores de Ocean Plastic')
calc_err(forecast_op,df_op)
print('\nErrores de Cardjolote Black')
calc_err(forecast_cb,df_cb)


# In[31]:


# Ventana móvil
df_cv = cross_validation(model, initial='540 days', period='30 days', horizon='90 days')

# Metricas reales fuera de muestra
df_p = performance_metrics(df_cv)
print(df_p[["horizon", "mae", "rmse", "mape", "coverage"]])


# In[22]:


# Ventana móvil
df_cv = cross_validation(model_black, initial='300 days', period='30 days', horizon='90 days')

# Metricas reales fuera de muestra
df_p = performance_metrics(df_cv)
print(df_p[["horizon", "mae", "rmse", "mape", "coverage"]])


# In[23]:


# Ventana móvil
df_cv = cross_validation(model_op, initial='260 days', period='30 days', horizon='100 days')

# Metricas reales fuera de muestra
df_p = performance_metrics(df_cv)
print(df_p[["horizon", "mae", "rmse", "mape", "coverage"]])


# In[24]:


# Ventana móvil
df_cv = cross_validation(model_op, initial='260 days', period='30 days', horizon='100 days')

# Metricas reales fuera de muestra
df_p = performance_metrics(df_cv)
print(df_p[["horizon", "mae", "rmse", "mape", "coverage"]])

