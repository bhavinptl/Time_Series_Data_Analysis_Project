#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller
# used for performing the augmented Dickey-Fuller test
from statsmodels.tsa.seasonal import seasonal_decompose
# used for decomposing a time series into its trend, seasonal, and residual components.
from statsmodels.tsa.stattools import acf, pacf
# used for analyzing the autocorrelation and partial autocorrelation of a time series.
from statsmodels.tsa.arima_model import ARIMA
from statsmodels import api as sm

get_ipython().system('pip install --upgrade statsmodels')


# In[2]:


airpass = pd.read_csv('C:/Users/bhavi/OneDrive/Desktop/AirPassengers.csv')
airpass
# This is time series data .
# where there is only one entity and given sequence of time.
# lets check some record


# In[3]:


airpass.info()


# In[15]:


# need to change the data type of month
from datetime import datetime
airpass['Month']=pd.to_datetime(airpass['Month'],infer_datetime_format=True)
airpass.info()


# In[16]:


# setting month column to an index
airpassind = airpass.set_index('Month',inplace=False)
airpassind.head()


# In[17]:


plt.xlabel('Date')
plt.ylabel('Number Of Air Passengers')
plt.plot(airpassind)


# In[18]:


def test_stationarity(timeseries):
    
    #Determine rolling statistics
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    
    #Plot rolling statistics
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(movingAverage, color='red', label='Rolling Mean')
    plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Augmented Dickey–Fuller test:
    print('Results of Dickey Fuller Test:')
    airpass_test = adfuller(airpassind['#Passengers'], autolag='AIC')
    dfoutput = pd.Series(airpass_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in airpass_test[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
   


# In[19]:


test_stationarity(airpassind)
#  based on the high p-value and the test statistic being greater than the critical values, 
#we fail to reject the null hypothesis, and we conclude that the time series is likely non-stationary.


# In[20]:


# Data Transformation To Achieve Stationarity
# Now, we will have to perform some data transformation to achieve Stationarity. 
# We can perform any of the transformations like taking 
# log scale, square, square root, cube, cube root, time shift, exponential decay etc.


# In[21]:


# Let's perform Log Transformation


# In[22]:


airpass_log = np.log(airpassind)
plt.plot(airpass_log)


# In[23]:


rollmean_log = airpass_log.rolling(window=12).mean()
rollstd_log = airpass_log.rolling(window=12).std()


# In[24]:


plt.plot(airpass_log, color='blue', label='Original')
plt.plot(rollmean_log, color='red', label='Rolling Mean')
plt.plot(rollstd_log, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation (Logarithmic Scale)')


# In[25]:


airpass_log_diff= airpass_log - rollmean_log


# In[26]:


# here we are dropping null values.
airpass_log_diff.dropna(inplace=True)
airpass_log_diff.head()


# In[27]:


# here we are using our pre defined function
test_stationarity(airpass_log_diff)


# In[28]:


# confidence intervals are pretty close to the Test Statistic
# we can now see that SD and Mean are constant in nature.
# so our model is Stationary now.


# In[29]:


decomposition = seasonal_decompose(airpass_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(airpass_log, label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# In[30]:


# There can be cases where an observation simply consist of trend & seasonality. 
# In that case, there won't be any residual component & that would be a null or NaN. 
# Hence, we also remove such cases.
residual = decomposition.resid
airpass_decompose = residual
airpass_decompose.dropna(inplace=True)
rollmean_decompose = airpass_decompose.rolling(window=12).mean()
rollstd_decompose = airpass_decompose.rolling(window=12).std()

plt.plot(airpass_decompose, color='blue', label='Original')
plt.plot(rollmean_decompose, color='red', label='Rolling Mean')
plt.plot(rollstd_decompose, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')


# In[31]:


lag_acf = acf(airpass_log_diff, nlags=20)
lag_pacf = pacf(airpass_log_diff, nlags=20, method='ols')


# In[32]:


#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(airpass_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(airpass_log_diff)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')            


#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(airpass_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(airpass_log_diff)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
            
plt.tight_layout()


# In[33]:


# plotting ACF and PCF in diffrent way for better understanding and better visualization
sm.graphics.tsa.plot_acf(airpass_log_diff,lags=20)
sm.graphics.tsa.plot_pacf(airpass_log_diff.squeeze(),lags=20,method='ols')
plt.show()


# In[34]:


from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Plotting the original logged passenger data
plt.plot(airpass_log, label='Original Data')

# Plotting the fitted values from the ARIMA model
plt.plot(results_AR.fittedvalues, color='red', label='Fitted Values')

# Setting the title of the plot to include the RSS value
plt.title('RSS: %.4f' % sum((results_AR.fittedvalues - airpass_log['Passengers'])**2))

# Displaying the legend
plt.legend()

# Displaying the plot
plt.show()

# Printing a message indicating the type of model being plotted
print('Plotting AR model')



# In[35]:


model2 = ARIMA(airpass_log, order=(0,1,2))
results_MA = model2.fit()
plt.plot(airpass_log)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_MA.fittedvalues - airpass_log['Passengers'])**2))
print('Plotting MA model')


# In[36]:


model = ARIMA(airpass_log_diff, order=(2, 1, 2))
results_ARIMA = model.fit()

plt.plot(airpass_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f' % sum((results_ARIMA.fittedvalues - airpass_log_diff) ** 2))
plt.show()


# In[37]:


model = ARIMA(airpass_log, order=(2,1,2))
results_ARIMA = model.fit()
plt.plot(airpass_log)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - airpass_log['Passengers'])**2))
print('Plotting ARIMA model')


# In[38]:


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.head()


# In[39]:


predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()


# In[40]:


predictions_ARIMA_log = pd.Series(airpass_log['#Passengers'].iloc[0], index=airpass_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()


# In[41]:


airpass_log.head()


# In[42]:


import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_predict

# Assuming you have already defined and trained the ARIMA model
model = ARIMA(airpass_log, order=(2, 1, 2))
results_ARIMA = model.fit()

plt.plot(airpass_log)
plot_predict(results_ARIMA, start=1, end=264)
plt.title('ARIMA Model - Forecast')
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.legend(['Observed', 'Forecast'])
plt.show()


# In[43]:


from datetime import datetime
airpass['Month']=pd.to_datetime(airpass['Month'],infer_datetime_format=True)


# In[44]:


airpass.head()


# In[62]:


airpass.tail()


# In[ ]:




