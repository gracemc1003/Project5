#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:38:03 2020

@author: gracemcmonagle
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt


filepath = '/Users/gracemcmonagle/Desktop/School/Fall 2020/EECS 731/Project 5/src/data/Historical Product Demand.csv'
rawData = pd.read_csv(filepath, delimiter = ',')
rawData['Date'] = pd.to_datetime(rawData['Date'], format='%Y/%m/%d')

#%%
# count the different unique products and warehouses, make sure no nan values
noProducts = rawData['Product_Code'].value_counts(dropna=False)
rawData['Warehouse'].value_counts(dropna=False)
rawData['Product_Category'].value_counts(dropna=False)

#%% We find that some of them are listed as (), so we need to fix that
#rawData['Order_Demand'] = rawData['Order_Demand'].astype(str)
#for index, row in rawData.iterrows():
#    row['Order_Demand'] = row['Order_Demand'].replace('(', '').replace(')', '')

#rawData['Order_Demand'] = pd.to_numeric(rawData['Order_Demand'])


#%%
specProd = rawData[rawData['Product_Code'] == 'Product_1341']
specProd.sort_values(by=['Date'])
#plt.hist(specProd['Order_Demand'])

specProd['Order_Demand'] = pd.to_numeric(specProd['Order_Demand'])
#plt.plot(specProd['Date'], specProd['Order_Demand'])

#no of orders on a specific day
specProd['Orders'] = specProd.groupby('Date')['Product_Code'].transform('count')
specProd['Demand'] = specProd.groupby('Date')['Order_Demand'].transform('sum')

#%%
#number of orders by month
noOrdersByDay = pd.DataFrame(specProd.groupby(specProd['Date'].dt.strftime('%Y/%m/%d'))['Orders'].sum())
demandByDay = pd.DataFrame(specProd.groupby(specProd['Date'].dt.strftime('%Y/%m/%d'))['Demand'].max())
noOrdersByDay['Day'] = noOrdersByDay.index
noOrdersByDay['Day'] = pd.to_datetime(noOrdersByDay['Day'], format='%Y/%m/%d')

byDay = pd.concat([noOrdersByDay, demandByDay], axis = 1)

byDay = byDay.sort_values(by=['Day'])
plt.scatter(byDay['Day'], byDay['Orders'])
plt.show()

plt.scatter(byDay['Day'], byDay['Demand'])

train, test = train_test_split(byDay['Demand'], shuffle=False)

model = SimpleExpSmoothing(np.asarray(byDay['Demand']))
model._index = pd.to_datetime(train.index)

fit1 = model.fit()
pred1 = fit1.forecast(test)
fit2 = model.fit(smoothing_level=.2)
pred2 = fit2.forecast(test)
fit3 = model.fit(smoothing_level=.5)
pred3 = fit3.forecast(test)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train.index[150:], train.values[150:])
ax.plot(test.index, test.values, color="gray")
for p, f, c in zip((pred1, pred2, pred3),(fit1, fit2, fit3),('#ff7823','#3c763d','c')):
    ax.plot(train.index[150:], f.fittedvalues[150:], color=c)
    ax.plot(test.index, p, label="alpha="+str(f.params['smoothing_level'])[:3], color=c)
plt.title("Simple Exponential Smoothing")    
plt.legend();
