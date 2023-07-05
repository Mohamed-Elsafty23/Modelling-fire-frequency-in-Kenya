# Script to fit and build different models
library(tidyverse)
library(MASS)
# Read in the data
analysis_data <- read_csv("fire_data_2000-18.csv")

# Check for overdispersion
summary(analysis_data$count) # mean
var(analysis_data$count) # variance

# We found, however, that there was over-dispersion in the data -
# the variance was larger than the mean in our dependent variable.

ggplot(analysis_data, aes(x=count)) + # plot a histogram
  geom_histogram(bins = 50)

# Split the data ensuring that the proportions are equal in both training (80%)
# and testing (20%) datasets
library(caret)
set.seed(3456)
trainIndex <- createDataPartition(analysis_data$count, p = .8, 
                                  list = FALSE, 
                                  times = 1)
# Create the datasets
fireTrain <- analysis_data[ trainIndex,]
fireTest  <- analysis_data[-trainIndex,]

#----------------------------------------------------------------
# Fit a poisson model
glmPoisson <- glm(count ~  mean_max_temp +
                       mean_rainfall,
                    data = fireTrain,
                    family = poisson(link = "log"))
summary(glmPoisson)

#Dont be fooled by the super significant coefficients. 
#Those beautiful p-values are exactly the consequences of the overdispersion
# issue. We can check the overdispersion either visually or quantitatively.

plot(log(fitted(glmPoisson)),
     log((fireTrain$count-fitted(glmPoisson))^2),
     xlab=expression(hat(mu)),
     ylab=expression((y-hat(mu))^2),pch=20,col="blue")
abline(0,1) ## 'varianc = mean' line

# We can see that the majority of the variance is larger than the mean,
# which is a warning of overdispersion.

# Calculate the overdispersion parameter
library(AER)
dispersiontest(glmPoisson) # overdispersion greater than 1 = 303.4566 

# We can check how much the coefficient estimations are affected by
# overdispersion.
summary(glmPoisson, dispersion = 303.4566)
#---------------------------------------------------------------
# A simple way to adjust the overdispersion is as straightforward as to
# estimate the dispersion parameter within the model. 
# This could be done via the quasi-families in R.

# Fit a quasipoisson model
glmQPoisson <- glm(count ~ mean_max_temp +
                     mean_rainfall,
                  data = fireTrain,
                  family = quasipoisson)
summary(glmQPoisson)

# We can see that the dispersion parameter is estimated to be 309.8148


#----------------------------------------------------------------
# NEGATIVE BINOMIAL

# Fit a negative binomial model
library(MASS)
# Version 1
glmNB <- glm.nb(count ~  mean_max_temp +
                  mean_rainfall, data = fireTrain,
                link = "log")

summary(glmNB)

# It is a better fit to the data because the ratio of deviance over degrees of freedom is only slightly larger than 1 here.

# Version 2

glmNB2 <- glm.nb(count ~ month + anomaly + mean_rainfall, data = fireTrain,
                link = "log")

summary(glmNB2)

# Version 3

glmNB3 <- glm.nb(count ~ month + average_temp + mean_rainfall,
                 data = fireTrain,
                 link = "log")

summary(glmNB2)

#The NB is more appropriate


# Simulate data
# count
set.seed(123)
y <- rnegbin(10000, mu = 277.1284, theta = 4.5)

# max temp
x1 <- rnorm(10000, mean = 29.18465, sd = 2.299063)

# min temp
x2 <- rnorm(10000, mean = 18.17286, sd = 2.13705)

# rainfall
x3 <- rnorm(10000, mean = 4.238619, sd = 0.6462258)
x3 <- exp(x3)

# make dataframe

sim1 <- data.frame(count = y,
                   max_temp = x1,
                   mean_temp = x2,
                   mean_pre = x3)


# Spatial negative binomial regression due to locations etc). 

library(lme4)

bin_glmm_fun = function(n_sims = 10000,
                        theta = 4.5,
                        y_mu = 277.1284,
                        x1_mu = 29.18465,
                        x1_sd = 2.299063,
                        x2_mu = 18.17286,
                        x2_sd = 2.13705,
                        x3_mu = 4.238619,
                        x3_sd = 0.6462258) {
  # set parameter
  
    # count
    y <- rnegbin(218, y_mu, theta = 1)
    
    # max temp
    x1 <- rnorm(218, x1_mu, x1_sd)
    
    # min temp
    x2 <- rnorm(218, x2_mu, x2_sd)
    
    # rainfall
    x3 <- rnorm(218, x3_mu, x3_sd)
    x3 <- exp(x3)
    
    # make dataframe
    
    sim1 <- data.frame(count = y,
                       month = analysis_data$month,
                       max_temp = x1,
                       min_temp = x2,
                       mean_prec = x3)
    # create training and test sets
    
    
    trainIndex <- createDataPartition(sim1$count, p = .8, 
                                      list = FALSE, 
                                      times = 1)
    # Create the datasets
    fireTrain <- sim1[ trainIndex,]
    fireTest  <- sim1[-trainIndex,]
    
    # Fit model on training set
    
    glmNB <- MASS::glm.nb(count ~ max_temp +
                      min_temp + mean_prec, data = fireTrain,
                    link = "log")
    # Predict on testing set
    predictions <- predict(glmNB,
                           newdata = fireTest, type = "response")
    
    test_rmse <- caret::RMSE(round(predictions),fireTest$count)
    # Dispersion parameter
    
    cbind(theta = glmNB$theta, test_rmse)
    
    
    
    
    
}

# Repeat the simulation many times
sims = replicate(1000, bin_glmm_fun(), simplify = FALSE )
sims[[100]]

#Explore estimated dispersion
library(purrr)

alldisp = map_df(sims, ~as.data.frame(.x), .id="id")

ggplot(alldisp, aes(x = theta) ) +
  geom_histogram(fill = "blue", 
                 alpha = .25, 
                 bins = 100) +
  geom_vline(xintercept = 1) +
  scale_x_continuous(breaks = seq(0, 2, by = 0.2) ) +
  theme_bw(base_size = 14) +
  labs(x = "Dispersion",
       y = "Count")
