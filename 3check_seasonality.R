# Script to check if real data has seasonality

# Read in the data
library(readr)
series_data <- read_csv("fire_data_2000-18.csv")

# Create max temp ts

tmax_ts <- ts(series_data$mean_max_temp,
              end = c(2018,12), frequency = 12)


# Create min temp ts
tmin_ts <- ts(series_data$mean_min_temp,
              end = c(2018,12), frequency = 12)

# Create rain ts
rain_ts <- ts(series_data$mean_rainfall,
              end = c(2018,12), frequency = 12)

# Check seasonality

# One approach is to use the tbats model, also in the forecast package in R. 
# It will handle seasonality and will automatically determine if a
# seasonal pattern is present.
# Then seasonal will be TRUE if a seasonal model is chosen and otherwise FALSE

library(forecast)
# Tmax
fit_tmax <- tbats(tmax_ts)
seasonal_tmax <- !is.null(fit_tmax$seasonal)
# Tmin
fit_tmin <- tbats(tmin_ts)
seasonal_tmin <- !is.null(fit_tmin$seasonal)
# Rainfall
fit_rain <- tbats(rain_ts)
seasonal_rain <- !is.null(fit_rain$seasonal)

# The data has seasonality