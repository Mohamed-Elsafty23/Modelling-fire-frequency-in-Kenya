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

# Source scripts
# source('simulation_temp.R')

# Generate data

# Set seed
# set.seed(3456)
df1=timeGad(nt=360)
# Add response variable
set.seed(678)
# Read in the data
analysis_data <- read_csv("fire_data_2000-18.csv")
# Fit NB
testnb <- glm.nb(count ~  mean_max_temp +
                   mean_min_temp + mean_rainfall, data = analysis_data)

# Generate random response variables with assumed seasonality
df1$count =  rnegbin(n = 360, theta = 4.5)

# Add time component
df1$time = rep(1:12, length.out = 360)
# Split data
library(caret)

trainIndex <- 0.8*360
# Create the datasets
fireTrain <- df1[1:288,]
fireTest  <- df1[289:360,]

# Fit NB model on train set
model1 <- glm.nb(count ~  max_temp +
                  min_temp + rainfall, data = fireTrain)

# Make predictions on normal NB
preds_nb <- round(exp(predict(model1, newdata = fireTest)),0)

############################################################

# Fit Bayesian Poisson MCMC model on train set
library(COUNT)
library(MCMCpack)
###########################################################################

# Fit Bayesian NB MCMC model
confint.default(model1)
m.bcnb <- MCMCnegbin(count ~  max_temp+
                        min_temp + rainfall,
                      data = fireTrain,
                      burnin = 1000, mcmc = 10000)



# Make predictions on test data MCMC Neg b

teb2 = summary(m.bcnb)
pRedr2 <- function(df,
                  int = teb2$statistics[,1][1],
                  x1 = teb2$statistics[,1][4],
                  x2 = teb2$statistics[,1][3],
                  x3 = teb2$statistics[,1][2]){
  pred <- exp(int + x1*df[,1] + x2*df[,2] + x3*df[,3])
  pred <- round(pred,0)
  pred
  
}

# Make predictions on test data (NB)
preds_bnb <- pRedr2(newtest)

#Model 3

# Fit Bayesian NB MCMC model
confint.default(model1)
m.bcnb2 <- MCMCnegbin(count ~  max_temp+
                       min_temp + rainfall + count_mean,
                     data = newtrain,
                     burnin = 1000, mcmc = 10000)



# Make predictions on test data MCMC Neg b

teb3 = summary(m.bcnb2)
pRedr3 <- function(df,
                   int = teb3$statistics[,1][1],
                   x1 = teb3$statistics[,1][4],
                   x2 = teb3$statistics[,1][3],
                   x3 = teb3$statistics[,1][2],
                   x4 = teb3$statistics[,1][5]){
  pred <- exp(int + x1*df[,1] + x2*df[,2] + x3*df[,3] + x4*df[,6])
  pred <- round(pred,0)
  pred
  
}

# Make predictions on test data (NB)
preds_bnbh <- pRedr3(newtest)

# Calculate RMSE (NB)
RMSE(pred = preds_nb, obs = fireTest$count)

# Calculate RMSE (BCMNB)
RMSE(pred = preds_bnb, obs = fireTest$count)

# Calculate RMSE (BNB2)
RMSE(pred = preds_bnbh, obs = fireTest$count)



##########################################################

# Fit PIG model
library(gamlss)
model_pg <- gamlss(count ~  max_temp +
                     min_temp + rainfall,
                   data = fireTrain, family=PIG)


















#####################################################################
# Fit stan model - POISSON and BAYESIAN POISSON
library(rstanarm)
poi1 <- stan_glm(count ~  max_temp+
                  min_temp + rainfall,
                data = fireTrain,
                family = poisson(link = "log"))

# Make prediction on poisson
preds_poi <- round(exp(predict(poi1, newdata = fireTest)),0)

# Check prior of poison
prior_summary(poi1)

# Implement on bayesian poisson

stanna_bp <- rstanarm::stan_glm(count ~  max_temp+
                               min_temp + rainfall,
                             data = fireTrain,
                             family = poisson(link = "log"),
                             prior = normal(location = c(0,0,0),
                                            scale = c(0.23,0.23,0.23)),
                             prior_intercept = normal(location = 0,
                                                      scale = 2.5))

# Make predictions
preds_bpoi <- posterior_predict(stanna_bp, newdata = fireTest)


# Find median on predictions
library(matrixStats)
preds_med <- data.frame(x = rowMedians(as.matrix(preds_bnb)))
preds_bnb <- data.frame(preds_bnb, median =preds_med) 
##########################################################################
# FITTING STANDARD NEGATIVE BINOMIAL MODEL
##########################################################################
negbin1 <- stan_glm.nb(count ~  max_temp+
                   min_temp + rainfall,
                 data = fireTrain,
                 link = "log")

# Make prediction on poisson
preds_negbin <- round(predict(negbin1, newdata = fireTest, type = "response"))

# Check prior
prior_summary(negbin1)

##########################################################################
# FITTING BAYESIAN NEGATIVE BINOMIAL MODEL
##########################################################################

# Implement on bayesian negative binomial

stanna_bnb <- rstanarm::stan_glm.nb(count ~  max_temp+
                               min_temp + rainfall,
                             data = fireTrain,
                             link = "log",
                             prior = normal(location = c(0,0,0),
                                            scale = c(0.27,0.26,0.23)),
                             prior_intercept = normal(location = 0,
                                                      scale = 2.5))
# Make predictions
set.seed(123)
post_pred <- posterior_predict(stanna_bnb,
                               newdata = fireTest,
                               draws = 1)
# Change predictions to data frame
library(purrr)
preds_bnb <- data.frame(count = t(data.frame(post_pred))) %>% 
  cbind(time = fireTest$time) %>% 
  inner_join(p_means, by = "time") %>% 
  rowwise() %>% 
  
  mutate(countt = rowMeans(across(starts_with("count")))) %>% 
  select(countt, time,  count_mean) 


# Add informative time component
# Get prior means

get_prior_means <- function(x){
  library(dplyr)
  x %>% group_by(time) %>% 
    summarize(count_mean = median(count)) %>% 
  data.frame()
}

p_means = get_prior_means(fireTrain)



##########################################################################
# FITTING BAYESIAN II NEGATIVE BINOMIAL MODEL
##########################################################################

# Join dataframe
newtrain <- fireTrain %>% inner_join(p_means, by = "time")
## MY MODEL
stanna_my <- rstanarm::stan_glm.nb(count ~  max_temp+
                                      min_temp + rainfall + count_mean,
                                    data = newtrain,
                                    link = "log",
                                    prior = normal(location = c(0,0,0,0),
                                                   scale = c(0.27,0.26,0.23,
                                                             1)),
                                    prior_intercept = normal(location = 0,
                                                             scale = 2.5))
# Make predictions
set.seed(123)
# Add new variable to train data
newtest <- fireTest %>% inner_join(p_means, by = "time")
post_pred2 <- posterior_predict(stanna_my,
                               newdata = newtest,
                               draws = 1)

preds_hyb2 <- data.frame(count = t(data.frame(post_pred2))) %>% 
  rowwise() %>% 
  
  mutate(countt = rowMeans(across(starts_with("count")))) 

# Check RMSE
library(Metrics)
# NB
RMSE(pred = preds_negbin, obs = fireTest$count)
mae(predicted =  preds_negbin, actual = fireTest$count)
mape(predicted = preds_negbin, actual = fireTest$count)
# BNB
RMSE(pred = preds_bnb$countt, obs = fireTest$count)
mae(predicted = preds_bnb$countt, actual = fireTest$count)
mape(predicted = preds_bnb$countt, actual = fireTest$count)

# TBNB-P
RMSE(pred = preds_hyb2$countt, obs = fireTest$count)
mae(predicted = preds_hyb2$countt, actual = fireTest$count)
mape(predicted = preds_hyb2$countt, actual = fireTest$count)



# BAYES 



