# Script to fit models on data
library(MASS)
library(stsm)
# Load libraries ----
library(rstanarm)
library(brms)  # for models
library(bayesplot)
library(ggplot2)
library(dplyr)
library(tidybayes)
library(modelr) 
#library(tidyverse)
library(caret)
library(readr)
library(purrr)
library(parallel)

##########################################################################
# FITTING STANDARD NEGATIVE BINOMIAL MODEL
##########################################################################

negbinner <- function(x, theta = 1.5, n = 60){
  
  # create training and test sets
  # set seed
  set.seed(456)
  
  
  trainIndex <- round(0.8*length(x$count))
  # Create the data sets
  fireTrain <- x[1:trainIndex,]
  fireTest  <- x[(trainIndex+1):n,]
  
  # Fit model on training set
  
  glmNB <- MASS::glm.nb(count ~ max_temp +
                          rainfall, data = fireTrain,
                        link = "log")
  # Predict on training set
  predictions_train <- predict(glmNB,
                         newdata = fireTrain, type = "response")
  # Predict on testing set
  predictions_test <- predict(glmNB,
                         newdata = fireTest, type = "response")
  # get the rmse
  train_rmse <- caret::RMSE(round(predictions_train),fireTrain$count)
  
  test_rmse <- caret::RMSE(round(predictions_test),fireTest$count)
  
  # get the MASE
  test_mase <- Metrics::mase(actual = fireTest$count,
                             predicted = round(predictions_test))
  
  # Get the bias
  test_bias <- Metrics::bias(actual = fireTest$count,
                             predicted = round(predictions_test))
  # Dispersion parameter
  
  cbind(rmse_train = train_rmse, rmse_test = test_rmse, mase_test = test_mase,
        bias_test = test_bias,
        theta = theta, n = n)
  
}
#########################################################################



##########################################################################
# FITTING BAYESIAN NEGATIVE BINOMIAL MODEL
##########################################################################

stanbinner <- function(x, theta = 1.5, n = 60){
  
  # create training and test sets
  # set seed
  set.seed(456)
  
  # Add time by month index
  x$time <- rep(1:12, length.out = n)
  
  
  trainIndex <- round(0.8*length(x$count))
  # Create the datasets
  fireTrain <- x[1:trainIndex,]
  fireTest  <- x[(trainIndex+1):n,]
  
  # Add prior means
  
  get_prior_means <- function(x){
    library(dplyr)
    x %>% group_by(time) %>% 
      summarize(count_mean = mean(count)) %>% 
      mutate(logsq = log(count_mean)) %>% 
      data.frame()
  }
  
  p_means = get_prior_means(fireTrain)
  
  # Join means to train data
  fireTrain2 <- fireTrain %>% 
    inner_join(p_means, by = "time")
  
  # Join means to test data
  fireTest2 <- fireTest %>% 
    inner_join(p_means, by = "time")
  
  
  # Fit model on training set
  
  stanNB <- rstanarm::stan_glm.nb(count ~  max_temp+
                                 rainfall + logsq,
                                 data = fireTrain2,
                                 link = "log")
  # Predict on training set
  predictions_train <- predict(stanNB,
                               newdata = fireTrain2, type = "response")
  # Predict on testing set
  predictions_test <- predict(stanNB,
                              newdata = fireTest2, type = "response")
  # get the rmse
  train_rmse <- caret::RMSE(round(predictions_train),fireTrain2$count)
  
  test_rmse <- caret::RMSE(round(predictions_test),fireTest2$count)
  
  # get the MASE
  test_mase <- Metrics::mase(actual = fireTest2$count,
                             predicted = round(predictions_test))
  # Get the bias
  test_bias <- Metrics::bias(actual = fireTest2$count,
                             predicted = round(predictions_test))
  # Dispersion parameter
  
  cbind(rmse_train = train_rmse, rmse_test = test_rmse, mase_test = test_mase,
        bias_test = test_bias,
        theta = theta, n = n)
  
}
##########################################################################

