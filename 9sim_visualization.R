# Script to read in and visualize simulation data
library(data.table)
library(tidyverse)

# NB Model
five_1.5 <- fread('./model_results/five_year_1.5_metrics.csv')
five_5 <- fread('./model_results/five_year_5_metrics.csv')
five_10 <- fread('./model_results/five_year_10_metrics.csv')
five_100 <- fread('./model_results/five_year_100_metrics.csv')
ten_1.5 <- fread('./model_results/ten_year_1.5_metrics.csv')
ten_5 <- fread('./model_results/ten_year_5_metrics.csv')
ten_10 <- fread('./model_results/ten_year_10_metrics.csv')
ten_100 <- fread('./model_results/ten_year_100_metrics.csv')
twenty_1.5 <- fread('./model_results/twenty_year_1.5_metrics.csv')
twenty_5 <- fread('./model_results/twenty_year_5_metrics.csv')
twenty_10 <- fread('./model_results/twenty_year_10_metrics.csv')
twenty_100 <- fread('./model_results/twenty_year_100_metrics.csv')
thirty_1.5 <- fread('./model_results/thirty_year_1.5_metrics.csv')
thirty_5 <- fread('./model_results/thirty_year_5_metrics.csv')
thirty_10 <- fread('./model_results/thirty_year_10_metrics.csv')
thirty_100 <- fread('./model_results/thirty_year_100_metrics.csv')

## Add model, theta
five_1.5$theta <- 1.5
five_5$theta <- 5
five_10$theta <- 10
five_100$theta <- 100
ten_1.5$theta <- 1.5
ten_5$theta <- 5
ten_10$theta <- 10
ten_100$theta <- 100
twenty_1.5$theta <- 1.5
twenty_5$theta <- 5
twenty_10$theta <- 10
twenty_100$theta <- 100
thirty_1.5$theta <- 1.5
thirty_5$theta <- 5
thirty_10$theta <- 10
thirty_100$theta <- 100
# combine files
nbmodels <- rbind(five_1.5, five_5, five_10, five_100, ten_1.5, ten_5, ten_10,
                  ten_100, twenty_1.5, twenty_5, twenty_10, twenty_100,
                  thirty_1.5, thirty_5,thirty_10, thirty_100)
# add model name
nbmodels$model <- "NB"



# BNB Model
five_1.5b <- fread('./model_results/five_year_1.5b_metrics.csv')
five_5b <- fread('./model_results/five_year_5b_metrics.csv')
five_10b <- fread('./model_results/five_year_10b_metrics.csv')
five_100b <- fread('./model_results/five_year_100b_metrics.csv')
ten_1.5b <- fread('./model_results/ten_year_1.5b_metrics.csv')
ten_5b <- fread('./model_results/ten_year_5b_metrics.csv')
ten_10b <- fread('./model_results/ten_year_10b_metrics.csv')
ten_100b <- fread('./model_results/ten_year_100b_metrics.csv')
twenty_1.5b <- fread('./model_results/twenty_year_1.5b_metrics.csv')
twenty_5b <- fread('./model_results/twenty_year_5b_metrics.csv')
twenty_10b <- fread('./model_results/twenty_year_10b_metrics.csv')
twenty_100b <- fread('./model_results/twenty_year_100b_metrics.csv')
thirty_1.5b <- fread('./model_results/thirty_year_1.5b_metrics.csv')
thirty_5b <- fread('./model_results/thirty_year_5b_metrics.csv')
thirty_10b <- fread('./model_results/thirty_year_10b_metrics.csv')
thirty_100b <- fread('./model_results/thirty_year_100b_metrics.csv')

## Add model, theta
five_1.5b$theta <- 1.5
five_5b$theta <- 5
five_10b$theta <- 10
five_100b$theta <- 100
ten_1.5b$theta <- 1.5
###########
ten_5b$theta <- 5
######
ten_10b$theta <- 10

############
ten_100b$theta <- 100
twenty_1.5b$theta <- 1.5
twenty_5b$theta <- 5
twenty_10b$theta <- 10
twenty_100b$theta <- 100
thirty_1.5b$theta <- 1.5
thirty_5b$theta <- 5
thirty_10b$theta <- 10
thirty_100b$theta <- 100
# combine files
bnbmodels <- rbind(five_1.5b, five_5b, five_10b, five_100b, ten_1.5b, ten_5b,
                   ten_10b,
                  ten_100b, twenty_1.5b, twenty_5b, twenty_10b, twenty_100b,
                  thirty_1.5b, thirty_5b,thirty_10b, thirty_100b)
# add model name
bnbmodels$model <- "BNB"

# Combine into one file
finalmetrics <- rbind(nbmodels, bnbmodels)



# Make plots by metrics

# RMSE on training set

plot1 <-  finalmetrics %>%
  ggplot(aes(x = as.factor(theta), y = rmse_train, fill = model))+
  geom_violin()+
  facet_wrap(~n,
             labeller = labeller(n = c("60" = "n = 60", 
                                           "120" = "n = 120", 
                                           "240" = "n = 240",
                                           "360" = "n = 360")))+
  # add labs
  labs(title = "c. RMSE on training data",
       #x = expression(paste("theta")),
       y = "RMSE on training data")+
  xlab(expression(paste("Theta ", theta)))+
  # add theme
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5, size = 16),
         axis.title.y = element_text(size = 16, face = "bold"),
         axis.title.x = element_text(size = 16, face = "bold"),
         axis.text.x = element_text(size = 16),
         axis.text.y = element_text(size = 16),
         legend.title = element_text(size = 16),
         legend.text = element_text(size = 16),
         strip.text = element_text(size = 16, face = "bold"))
plot1

# RMSE on test datasets
plot2 <-  finalmetrics %>%
  ggplot(aes(x = as.factor(theta), y = rmse_test, fill = model))+
  geom_violin()+
  facet_wrap(~n,
             labeller = labeller(n = c("60" = "n = 60", 
                                       "120" = "n = 120", 
                                       "240" = "n = 240",
                                       "360" = "n = 360")))+
  # add labs
  labs(title = "d. RMSE on test data",
       #x = expression(paste("theta")),
       y = "RMSE on test data")+
  xlab(expression(paste("Theta ", theta)))+
  # add theme
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title.y = element_text(size = 16, face = "bold"),
        axis.title.x = element_text(size = 16, face = "bold"),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 16),
        strip.text = element_text(size = 16, face = "bold"))

# BIAS on testing set

plot3 <- finalmetrics %>%
  ggplot(aes(x = as.factor(theta), y = bias_test, fill = model))+
  geom_violin()+
  facet_wrap(~n,
             labeller = labeller(n = c("60" = "n = 60", 
                                       "120" = "n = 120", 
                                       "240" = "n = 240",
                                       "360" = "n = 360")))+
  # add labs
  labs(title = "a. Bias on test data",
       #x = expression(paste("theta")),
       y = "Bias on test data")+
  xlab(expression(paste("Theta ", theta)))+
  # add theme
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title.y = element_text(size = 16, face = "bold"),
        axis.title.x = element_text(size = 16, face = "bold"),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 16),
        strip.text = element_text(size = 16, face = "bold"))

# MASE on testing set

plot4 <- finalmetrics %>%
  ggplot(aes(x = as.factor(theta), y = mase_test, fill = model))+
  geom_violin()+
  facet_wrap(~n,
             labeller = labeller(n = c("60" = "n = 60", 
                                       "120" = "n = 120", 
                                       "240" = "n = 240",
                                       "360" = "n = 360")))+
  # add labs
  labs(title = "b. MASE on test data",
       #x = expression(paste("theta")),
       y = "MASE on test data")+
  xlab(expression(paste("Theta ", theta)))+
  # add theme
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title.y = element_text(size = 16, face = "bold"),
        axis.title.x = element_text(size = 16, face = "bold"),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 16),
        strip.text = element_text(size = 16, face = "bold"))

# Combine the four graphs
library(gridExtra)
comp_plot <- grid.arrange(plot3,# bias
                          plot4, # mase
                          plot1, # rmse1
                          plot2, # rmse2
                          name = "Model performance") 






# Visualize model
library(tidyverse)

finalmet2 = finalmetrics %>%
  pivot_longer(rmse_train:bias_test, names_to = "metric", values_to = "value") 

finalmet2[mapply(is.infinite, finalmet2)] <- NA
# BIAS
bias1 = 
finalmet2 %>% 
  filter(metric == "bias_test") %>% 
  group_by(model,theta, n) %>% 
  summarize(average = mean(value, na.rm = T)) %>% pivot_wider(
                                                              values_from = "average",
                                                              names_from = "model") %>% 
  rename(BNB_bias = BNB, NB_bias = NB)%>% ungroup()
# MASE
mase1 = 
finalmet2 %>% 
  filter(metric == "mase_test") %>% 
  group_by(model,theta, n) %>% 
  summarize(average = mean(value, na.rm = T)) %>% pivot_wider(
    values_from = "average",
    names_from = "model") %>% 
  rename(BNB_mase = BNB, NB_mase = NB)%>% ungroup()

# RMSE TEST
rmse2=
finalmet2 %>% 
  filter(metric == "rmse_test") %>% 
  group_by(model,theta, n) %>% 
  summarize(average = mean(value, na.rm = T)) %>% pivot_wider(
    values_from = "average",
    names_from = "model") %>% 
  rename(BNB_rmse2 = BNB, NB_rmse2 = NB)%>% ungroup()

# RMSE TRAIN
rmse1 =
finalmet2 %>% 
  filter(metric == "rmse_train") %>% 
  group_by(model,theta, n) %>% 
  summarize(average = mean(value, na.rm = T)) %>% pivot_wider(
    values_from = "average",
    names_from = "model")%>% 
  rename(BNB_rmse1 = BNB, NB_rmse1 = NB) %>% ungroup()

# Combine all results
sim_results = cbind(bias1, mase1[,-c(1:2)],
                    rmse1[,-c(1:2)],
                    rmse2[,-c(1:2)])
