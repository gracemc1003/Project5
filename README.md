# Project 5

In this project, we used warehouse data to attempt to predict the future demand for a certain product based on its past demand. I approached this problem by training different models on a given product's data, and then creating a forecast for that certain product.

Some of the dataset modifications were:
* A noOrders variable to count the number of orders on a certain day.
* Analyzing which variables actually added information to the model

## Forecasting Methods
We attempted to run two different regression methods for our predictor: exponential smoothing and ARIMA.

### Exponential Smoothing
In this model, we trained an exponential smoothing model with two different $\alpha$ values. We found that a higher $\alpha$ value helped to match the data with peaks and spikes. In both cases though, the difference in $\alpha$ didn't drastically change the forecast, though. 

### ARIMA

In the ARIMA model, we measured its success by measuring the residuals of the model. We found that our residuals were around 0 which is what we'd hope to see. So, we can use this model to make a forecast for the demand of the product in the future.
