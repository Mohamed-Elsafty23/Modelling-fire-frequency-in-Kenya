# Simulate climate data
library(MASS)
library(fitdistrplus)
# Assume count data is Negative Binomial distribution

# Temperature data is from a normal distribution

# Use the fitdistr function to estimate the shape parameters
# of a weibull distribution that best fit the rainfall data
library(readr)
datak <- read_csv('fire_data_2000-18.csv')
fit_x1 <- fitdist(datak$mean_rainfall, "gamma")
fit_x2 <- fitdist(datak$mean_max_temp, "norm")

# The estimated shape parameters are stored in the "estimate" field of the fit object
lambda = fit_x1$estimate[1]
k = fit_x2$estimate[2]

# select columns
datak <- datak[,c(8,10,4)]
# change names
colnames(datak) <- c("max_temp","rainfall", "count")

# add tyme column
datak$tyme = 1:218



# fit a linear model
lmodel = lm(rainfall ~ max_temp, data = datak)

#-----------------------------------------------------
# Generate time series
timeGad <- 
  function(nt = 24, thet = 1.5, mod = lmodel, data = datak){
    
    # Load the mnormt library
    library(mnormt)
    
    # Set the mean, covariance matrix, and correlations
    
    # Generate the simulated data
    #rain <- rlnorm(n = nt, meanlog = 4.2386185, 
     #              sdlog = 0.6447419)
    #rain <- rgamma(n = nt, shape = 2.74965460, 
     #              rate = 0.03271982)
    # fit a model
    fm  <- MASS::glm.nb(count ~ max_temp +
                          rainfall +
                          #tcos,
                          sin((2*12*pi/tyme) +rnorm(1,sd=0.1))+
                          cos((2*12*pi/tyme) + rnorm(1,sd=0.1)),
                        data = datak,
                        link = "log",
                        init.theta = thet,
                        control = glm.control(maxit = 100))
    
    max_temp = TruncatedNormal::rtnorm(n = nt, mu = 29.184653,
                     sd = 2.293784, lb = 23.43933,
                     ub = 34.8166)
    
    #------------------------------------------------------------
    # Combine all data
    dataclean = data.frame(max_temp)
    #####################
    # Calculate the residuals
    resid <- signif(residuals(mod), 5)
    
    # and the predicted values
    predicted_values <- predict(mod, newdata = dataclean)
    
    # Set the bounds
    bounds <- seq(from = min(resid), to = max(resid), by = 0.2)
    # Check the relative frequencies and error distribution
    error_distribution <- table(cut(resid, bounds))/sum(table(cut(resid,bounds)))
    # We now generate 10 errors
    simulated_errors <- runif(length(predicted_values), min = min(resid), max = max(resid))
    simulated_values <- c()
    
    # We simulate actual values of Y by adding errors to the predicted values
    for(i in 1:length(dataclean$max_temp))
    {
      simulated_values[i] <- (predicted_values[i] + simulated_errors[i])
    }
    
    
    #####################
    # Add rain
    #dataclean$r = 7.835*(dataclean$max_temp)
    #dataclean$rain = rgamma(n = nt, rate = 0.03271982 ,
    #                            shape = 2.74965460)
    # add randomness
    #dataclean$teli = runif(nt, max = 0.03271982)
    dataclean$rainfall = simulated_values
    
    dataclean$tyme = 1:nt
    
    ######################################
    
    
    #y_count <- round(predict(modu, newdata = dataclean, type = "response"))
    
    
    #y_count2 <- rnegbin(n = nt, mu = 277.1284, theta = 1.5)
    y_count2 <- predict(fm, dataclean, type = "response")
    # Calculate the residuals
    resid2 <- signif(residuals(fm), 5)
    
    # We now generate 10 errors
    simulated_errors2 <- runif(length(y_count2), min = min(resid2), max = max(resid2))
    simulated_values2 <- c()
    
    # We simulate actual values of Y by adding errors to the predicted values
    for(i in 1:nt)
    {
      simulated_values2[i] <- (y_count2[i] + simulated_errors2[i])
    }
    
    # Here are our simulated_values
    # Result data
    dataclean$count <- round(simulated_values2)
    
    dataclean
  }


# Simulate climate data

temper <- 
  function(n = 10, n2 = 60, theta = 1.5, pathway = "simulated_data/"){
    # Set empty list
    daf = list()
    
    for(i in 1:n){
      # create data
      df <- timeGad(nt=n2, thet = theta, mod = lmodel, data = datak)
      daf[[i]] <- df}
    # Write out list as csv files
    for(i in 1:length(daf)){
      readr::write_csv(data.frame(daf[[i]]),
                file = paste0(pathway,"dyear", n2/12,"_",theta,"_",i, '.csv'))
    }
    
  }

# Generate data (theta = 1.5,)
## 5 year data
set.seed(76568)
temper(n=1000, n2 = 60, theta = 1.5, pathway = "simulated_data/d5year/theta_1.5/")
beepr::beep(sound = 1)
## 10 year data
temper(n=1000, n2 = 120, theta = 1.5, pathway = "simulated_data/d10year/theta_1.5/")
beepr::beep(sound = 1)
## 20 year data
temper(n=1000, n2 = 240, theta = 1.5, pathway = "simulated_data/d20year/theta_1.5/")
beepr::beep(sound = 1)
## 30 year data
temper(n=1000, n2 = 360, theta = 1.5,pathway = "simulated_data/d30year/theta_1.5/")
beepr::beep(sound = 1)
# Generate data (theta = 5)
## 5 year data
set.seed(76568)
temper(n=1000, n2 = 60, theta = 5, pathway = "simulated_data/d5year/theta_5/")
beepr::beep(sound = 1)
## 10 year data
temper(n=1000, n2 = 120, theta = 5, pathway = "simulated_data/d10year/theta_5/")
beepr::beep(sound = 1)
## 20 year data
temper(n=1000, n2 = 240, theta = 5, pathway = "simulated_data/d20year/theta_5/")
beepr::beep(sound = 1)
## 30 year data
temper(n=1000, n2 = 360,  theta = 5,pathway = "simulated_data/d30year/theta_5/")
beepr::beep(sound = 1)
# Generate data (theta = 10)
## 5 year data
set.seed(76568)
temper(n=1000, n2 = 60, theta = 10, pathway = "simulated_data/d5year/theta_10/")
beepr::beep(sound = 1)
## 10 year data
temper(n=1000, n2 = 120, theta = 10, pathway = "simulated_data/d10year/theta_10/")

## 20 year data
temper(n=1000, n2 = 240, theta = 10, pathway = "simulated_data/d20year/theta_10/")

## 30 year data
temper(n=1000, n2 = 360, theta = 10, pathway = "simulated_data/d30year/theta_10/")

# Generate data (theta = 100)
## 5 year data
set.seed(76568)
temper(n=1000, n2 = 60, theta = 100, pathway = "simulated_data/d5year/theta_100/")

## 10 year data
temper(n=1000, n2 = 120, theta = 100, pathway = "simulated_data/d10year/theta_100/")

## 20 year data
temper(n=1000, n2 = 240, theta = 100, pathway = "simulated_data/d20year/theta_100/")

## 30 year data
temper(n=1000, n2 = 360, theta = 100, pathway = "simulated_data/d30year/theta_100/")

