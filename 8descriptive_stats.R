# Script to generate descriptive statistics and graphs


## --------------------------------------------------------------------------------------------------------------------
# Script to start out the results section
library(tidyverse)
# read data
fire_clim <- read_csv('fire_data_2000-18.csv')



## --------------------------------------------------------------------------------------------------------------------
# Find the summary stats
library(rstatix)
library(flextable)
fire_clim2 <-  fire_clim[,c(4,8:10)]

fire_clim2 %>% 
  get_summary_stats()


## --------------------------------------------------------------------------------------------------------------------
# number of fires
## Create a time series object
count_ts <- fire_clim |> 
  mutate(Time = paste(year,month,sep = "-")) |> 
  mutate(Time = zoo::as.yearmon(Time))


## --------------------------------------------------------------------------------------------------------------------
# Plot the time series
plot_count <- 
ggplot(count_ts, aes(Time, count )) +  geom_line(col = "red") +
  scale_x_continuous(breaks = seq(2000,2018,5)) + theme_bw() +
  labs(title = "a. Monthly fire frequency trend from \n2000 to 2018 in Kenya")+
  ylab("Number of fires")+
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title.y = element_text(size = 16, face = "bold"),
        axis.title.x = element_text(size = 16, face = "bold"),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16))

plot_count



## --------------------------------------------------------------------------------------------------------------------
# Plot bars
plot_count2 <- 
  count_ts %>% 
  group_by(month) %>% 
  summarize(mean_count = mean(count)) %>% 
  ggplot(aes(x = month, y = mean_count )) +  geom_bar(col = "red", stat = "identity") +
  #scale_x_continuous(breaks = seq(2000,2018,5)) + theme_bw() +
  labs(title = "a. Monthly fire frequency trend from 2000 to 2018 in Kenya")+
  ylab("Number of fires")+
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.y = element_text(size = 14),
        axis.title.x = element_text(size = 14),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
        legend.position = "none")

plot_count2



## --------------------------------------------------------------------------------------------------------------------
# rainfall and fire frequency
corplot1 <- 
count_ts %>% 
  mutate(quarteR = ifelse(month < 4, "Qrt1", ifelse(month > 3 & month < 7, "Qrt2",
                                                    ifelse(month > 9, "Qrt4", "Qrt3")))) %>% 
  ggplot(aes(x = mean_rainfall, y = log(count)))+
  # add points
  geom_point() +
  # add regression line
  geom_smooth(formula = 'y~x', method = 'lm')+
  theme_minimal()+
  labs(title = "Fire frequency vs maximum temperature",
       x = "Rainfall (mm)",
       y =  "logarithm of fire frequency") +
  facet_wrap( ~quarteR, scales = "free")
corplot1


## --------------------------------------------------------------------------------------------------------------------
# max temperature

# Plot the time series
plot_max <- 
ggplot(count_ts, aes(Time, mean_max_temp)) +  geom_line( col = "brown") +
  scale_x_continuous(breaks = seq(2000,2018,5)) + theme_bw() +
  labs(title = "b. Monthly maximum temperature trend \nfrom 2000 to 2018 in Kenya")+
  ylab("Max temperature (\u00B0C)")+
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title.y = element_text(size = 16, face = "bold"),
        axis.title.x = element_text(size = 16, face = "bold"),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16))
plot_max


## --------------------------------------------------------------------------------------------------------------------
# min temperature

# Plot the time series
plot_min <- 
ggplot(count_ts, aes(Time, mean_min_temp)) +  geom_line(col = "blue") +
  scale_x_continuous(breaks = seq(2000,2018,5)) + theme_bw() +
  labs(title = "c. Monthly minimum temperature trend \nfrom 2000 to 2018 in Kenya")+
  ylab("Min temperature (\u00B0C)")+
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title.y = element_text(size = 16, face = "bold"),
        axis.title.x = element_text(size = 16, face = "bold"),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16))
plot_min


## --------------------------------------------------------------------------------------------------------------------
# min temperature and fire frequency
corplot2 <- 
count_ts %>% 
  mutate(quarteR = ifelse(month < 4, "Qrt1", ifelse(month > 3 & month < 7, "Qrt2",
                                                    ifelse(month > 9, "Qrt4", "Qrt3")))) %>% 
  ggplot(aes(x = mean_min_temp, y = log(count)))+
  # add points
  geom_point() +
  # add regression line
  geom_smooth(formula = 'y~x', method = 'lm')+
  theme_minimal()+
  labs(title = "Fire frequency vs minimum temperature",
       x = "Mean minimum temperature (°C)",
       y =  "Fire frequency") +
  facet_wrap( ~quarteR, scales = "free")
corplot2


## --------------------------------------------------------------------------------------------------------------------
# max temperature and fire frequency
corplot3 <- 
count_ts %>% 
  mutate(quarteR = ifelse(month < 4, "Qrt1", ifelse(month > 3 & month < 7, "Qrt2",
                                                    ifelse(month > 9, "Qrt4", "Qrt3")))) %>% 
  ggplot(aes(x = mean_max_temp, y = log(count)))+
  # add points
  geom_point() +
  # add regression line
  geom_smooth(formula = 'y~x', method = 'lm')+
  theme_minimal()+
  labs(title = "Fire frequency vs maximum temperature",
       x = "Mean maximum temperature (°C)",
       y =  "Fire frequency") +
  facet_wrap( ~quarteR, scales = "free")
corplot3


## --------------------------------------------------------------------------------------------------------------------
# rainfall

# Plot the time series
plot_rain <- 
ggplot(count_ts, aes(Time, mean_rainfall)) +  geom_line(col = "orange") +
  scale_x_continuous(breaks = seq(2000,2018,5)) + theme_bw() +
  labs(title = "d. Monthly rainfall trend from \n2000 to 2018 in Kenya")+
  ylab("Rainfall (mm)")+
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title.y = element_text(size = 16, face = "bold"),
        axis.title.x = element_text(size = 16, face = "bold"),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16))

plot_rain


## ---- fig.width=8, fig.height=8--------------------------------------------------------------------------------------
# Combine the plots
library(patchwork)
(plot_count + plot_max) / (plot_min + plot_rain)


## --------------------------------------------------------------------------------------------------------------------
# Read in the four time periods
five_a <- read_csv("./model_results/five_year_1.5_metrics.csv")
# add mdtype column
five_a$model <- "NB"

five_b <- read_csv("./model_results/five_year_1.5b_metrics.csv")
# add mdtype
five_b$model <- "BNB"
# combine the two
five_pnt1.5 <- rbind(five_a, five_b)
head(five_pnt1.5)


## ---- fig.width=8, fig.height=6--------------------------------------------------------------------------------------
# Long format
library(tidyr)
five_pnt1.5 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  # Plot line graphs
  ggplot(aes(y = value, x = ...1, group = model, col = model))+
  geom_line() +
  facet_wrap(~ metric, scales = "free") +
  # add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size n = 60,  \u03B8 = 1.5",
       x = "Dataset number")


## --------------------------------------------------------------------------------------------------------------------
library(flextable)
s1_60 <- 
five_pnt1.5 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  group_by(metric,model) %>% 
  summarise(mean_val = mean(value)) %>% 
  spread(key = model, value = mean_val) %>% 
  mutate(n = 60, theta = 1.5)

# Make table
s1_60 %>% 
  mutate_if(is.numeric,round, 2) %>% 
  regulartable()


## --------------------------------------------------------------------------------------------------------------------
# Read in the four time periods
ten_a <- read_csv("./model_results/ten_year_1.5_metrics.csv")
# add mdtype column
ten_a$model <- "NB"

ten_b <- read_csv("./model_results/ten_year_1.5b_metrics.csv")
# add mdtype
ten_b$model <- "BNB"
# combine the two
ten_pnt1.5 <- rbind(ten_a, ten_b)
head(ten_pnt1.5)


## ---- fig.width=8, fig.height=6--------------------------------------------------------------------------------------
# Long format
library(tidyr)
ten_pnt1.5 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  # Plot line graphs
  ggplot(aes(y = value, x = ...1, group = model, col = model))+
  geom_line() +
  facet_wrap(~ metric, scales = "free") +
  # add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size n = 120,  \u03B8 = 1.5",
       x = "Dataset number")


## --------------------------------------------------------------------------------------------------------------------
library(flextable)
s1_120 <- 
ten_pnt1.5 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  group_by(metric,model) %>% 
  summarise(mean_val = mean(value)) %>% 
  spread(key = model, value = mean_val)  %>% 
  mutate(n = 120, theta = 1.5)
# make table
s1_120 %>% 
  mutate_if(is.numeric,round, 2) %>% 
  regulartable()


## --------------------------------------------------------------------------------------------------------------------
# Read in the four time periods
twenty_a <- read_csv("./model_results/twenty_year_1.5_metrics.csv")
# add mdtype column
twenty_a$model <- "NB"

twenty_b <- read_csv("./model_results/twenty_year_1.5b_metrics.csv")
# add mdtype
twenty_b$model <- "BNB"
# combine the two
twenty_pnt1.5 <- rbind(twenty_a, twenty_b)
head(twenty_pnt1.5)


## ---- fig.width=8, fig.height=6--------------------------------------------------------------------------------------
# Long format
library(tidyr)
twenty_pnt1.5 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  # Plot line graphs
  ggplot(aes(y = value, x = ...1, group = model, col = model))+
  geom_line() +
  facet_wrap(~ metric, scales = "free") +
  # add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size n = 240,  \u03B8 = 1.5",
       x = "Dataset number")


## --------------------------------------------------------------------------------------------------------------------
library(flextable)

s1_240 <- 
twenty_pnt1.5 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  group_by(metric,model) %>% 
  summarise(mean_val = mean(value)) %>% 
  spread(key = model, value = mean_val)  %>% 
  mutate(n = 240, theta = 1.5)


s1_240 %>% 
  mutate_if(is.numeric,round, 2) %>% 
  regulartable()


## --------------------------------------------------------------------------------------------------------------------
# Read in the four time periods
thirty_a <- read_csv("./model_results/thirty_year_1.5_metrics.csv")
# add mdtype column
thirty_a$model <- "NB"

thirty_b <- read_csv("./model_results/thirty_year_1.5b_metrics.csv")
# add mdtype
thirty_b$model <- "BNB"
# combine the two
thirty_pnt1.5 <- rbind(thirty_a, thirty_b)
head(thirty_pnt1.5)


## ---- fig.width=8, fig.height=6--------------------------------------------------------------------------------------
# Long format
library(tidyr)
thirty_pnt1.5 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  # Plot line graphs
  ggplot(aes(y = value, x = ...1, group = model, col = model))+
  geom_line() +
  facet_wrap(~ metric, scales = "free") +
  # add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size n = 360,  \u03B8 = 1.5",
       x = "Dataset number")


## --------------------------------------------------------------------------------------------------------------------
library(flextable)
s1_360 <- 
thirty_pnt1.5 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  group_by(metric,model) %>% 
  summarise(mean_val = mean(value)) %>% 
  spread(key = model, value = mean_val)  %>% 
  mutate(n = 360, theta = 1.5)


s1_360 %>% 
  mutate_if(is.numeric,round, 2) %>% 
  regulartable()


## --------------------------------------------------------------------------------------------------------------------
# Read in the four time periods
five_a5 <- read_csv("./model_results/five_year_5_metrics.csv")
# add mdtype column
five_a5$model <- "NB"

five_b5 <- read_csv("./model_results/five_year_5b_metrics.csv")
# add mdtype
five_b5$model <- "BNB"
# combine the two
five_pnt5 <- rbind(five_a5, five_b5)
head(five_pnt5)


## ---- fig.width=8, fig.height=6--------------------------------------------------------------------------------------
# Long format
library(tidyr)
five_pnt5 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  # Plot line graphs
  ggplot(aes(y = value, x = ...1, group = model, col = model))+
  geom_line() +
  facet_wrap(~ metric, scales = "free") +
  # add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size n = 60, \u03B8 = 5",
       x = "Dataset number")


## --------------------------------------------------------------------------------------------------------------------
library(flextable)
s2_60 <- 
five_pnt5 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  group_by(metric,model) %>% 
  summarise(mean_val = mean(value)) %>% 
  spread(key = model, value = mean_val)  %>% 
  mutate(n = 60, theta = 5)


s2_60 %>% 
  mutate_if(is.numeric,round, 2) %>% 
  regulartable()


## --------------------------------------------------------------------------------------------------------------------
# Read in the four time periods
ten_a5 <- read_csv("./model_results/ten_year_5_metrics.csv")
# add mdtype column
ten_a5$model <- "NB"

ten_b5 <- read_csv("./model_results/ten_year_5b_metrics.csv")
# add mdtype
ten_b5$model <- "BNB"
# combine the two
ten_pnt5 <- rbind(ten_a5, ten_b5)
head(ten_pnt5)


## ---- fig.width=8, fig.height=6--------------------------------------------------------------------------------------
# Long format
library(tidyr)
ten_pnt5 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  # Plot line graphs
  ggplot(aes(y = value, x = ...1, group = model, col = model))+
  geom_line() +
  facet_wrap(~ metric, scales = "free") +
  # add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size n = 120, \u03B8 = 5",
       x = "Dataset number")


## --------------------------------------------------------------------------------------------------------------------
library(flextable)
s2_120 <- 
ten_pnt5 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  group_by(metric,model) %>% 
  summarise(mean_val = mean(value)) %>% 
  spread(key = model, value = mean_val)  %>% 
  mutate(n = 120, theta = 5)


s2_120 %>% 
  mutate_if(is.numeric,round, 2) %>% 
  regulartable()


## --------------------------------------------------------------------------------------------------------------------
# Read in the four time periods
twenty_a5 <- read_csv("./model_results/twenty_year_5_metrics.csv")
# add mdtype column
twenty_a5$model <- "NB"

twenty_b5 <- read_csv("./model_results/twenty_year_5b_metrics.csv")
# add mdtype
twenty_b5$model <- "BNB"
# combine the two
twenty_pnt5 <- rbind(twenty_a5, twenty_b5)
head(twenty_pnt5)


## ---- fig.width=8, fig.height=6--------------------------------------------------------------------------------------
# Long format
library(tidyr)
twenty_pnt5 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  # Plot line graphs
  ggplot(aes(y = value, x = ...1, group = model, col = model))+
  geom_line() +
  facet_wrap(~ metric, scales = "free") +
  # add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size n = 240, \u03B8 = 5",
       x = "Dataset number")


## --------------------------------------------------------------------------------------------------------------------
library(flextable)
s2_240 <- 
twenty_pnt5 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  group_by(metric,model) %>% 
  summarise(mean_val = mean(value)) %>% 
  spread(key = model, value = mean_val) %>% 
  mutate(n = 240, theta = 5)



s2_240 %>% 
  mutate_if(is.numeric,round, 2) %>% 
  regulartable()


## --------------------------------------------------------------------------------------------------------------------
# Read in the four time periods
thirty_a5 <- read_csv("./model_results/thirty_year_5_metrics.csv")
# add mdtype column
thirty_a5$model <- "NB"

thirty_b5 <- read_csv("./model_results/thirty_year_5b_metrics.csv")
# add mdtype
thirty_b5$model <- "BNB"
# combine the two
thirty_pnt5 <- rbind(thirty_a5, thirty_b5)
head(thirty_pnt5)


## ---- fig.width=8, fig.height=6--------------------------------------------------------------------------------------
# Long format
library(tidyr)
thirty_pnt5 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  # Plot line graphs
  ggplot(aes(y = value, x = ...1, group = model, col = model))+
  geom_line() +
  facet_wrap(~ metric, scales = "free") +
  # add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard \nNegative Binomial model",
       subtitle = "Comparison of four metrics with sample size n = 360, \u03B8 = 5",
       x = "Dataset number")


## --------------------------------------------------------------------------------------------------------------------
library(flextable)
s2_360 <- 
thirty_pnt5 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  group_by(metric,model) %>% 
  summarise(mean_val = mean(value)) %>% 
  spread(key = model, value = mean_val) %>% 
  mutate(n = 360, theta = 5)



s2_360 %>% 
  mutate_if(is.numeric,round, 2) %>% 
  regulartable()


## --------------------------------------------------------------------------------------------------------------------
# Read in the four time periods
five_a10 <- read_csv("./model_results/five_year_10_metrics.csv")
# add mdtype column
five_a10$model <- "NB"

five_b10 <- read_csv("./model_results/five_year_10b_metrics.csv")
# add mdtype
five_b10$model <- "BNB"
# combine the two
five_pnt10 <- rbind(five_a10, five_b10)
head(five_pnt10)


## ---- fig.width=8, fig.height=6--------------------------------------------------------------------------------------
# Long format
library(tidyr)
five_pnt10 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  # Plot line graphs
  ggplot(aes(y = value, x = ...1, group = model, col = model))+
  geom_line() +
  facet_wrap(~ metric, scales = "free") +
  # add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size n = 60, \u03B8 = 10",
       x = "Dataset number")


## --------------------------------------------------------------------------------------------------------------------
library(flextable)

s3_60 <- 
five_pnt10 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  group_by(metric,model) %>% 
  summarise(mean_val = mean(value)) %>% 
  spread(key = model, value = mean_val) %>% 
  mutate(n = 60, theta = 10)


s3_60 %>% 
  mutate_if(is.numeric,round, 2) %>% 
  regulartable()


## --------------------------------------------------------------------------------------------------------------------
# Read in the four time periods
ten_a10 <- read_csv("./model_results/ten_year_10_metrics.csv")
# add mdtype column
ten_a10$model <- "NB"

ten_b10 <- read_csv("./model_results/ten_year_10b_metrics.csv")
# add mdtype
ten_b10$model <- "BNB"
# combine the two
ten_pnt10 <- rbind(ten_a10, ten_b10)
head(ten_pnt10)


## ---- fig.width=8, fig.height=6--------------------------------------------------------------------------------------
# Long format
library(tidyr)
ten_pnt10 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  # Plot line graphs
  ggplot(aes(y = value, x = ...1, group = model, col = model))+
  geom_line() +
  facet_wrap(~ metric, scales = "free") +
  # add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size n = 120, \u03B8 = 10",
       x = "Dataset number")


## --------------------------------------------------------------------------------------------------------------------
library(flextable)

s3_120 <- 
ten_pnt10 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  group_by(metric,model) %>% 
  summarise(mean_val = mean(value)) %>% 
  spread(key = model, value = mean_val) %>% 
  mutate(n = 120, theta = 10)


s3_120 %>% 
  mutate_if(is.numeric,round, 2) %>% 
  regulartable()


## --------------------------------------------------------------------------------------------------------------------
# Read in the four time periods
twenty_a10 <- read_csv("./model_results/twenty_year_10_metrics.csv")
# add mdtype column
twenty_a10$model <- "NB"

twenty_b10 <- read_csv("./model_results/twenty_year_10b_metrics.csv")
# add mdtype
twenty_b10$model <- "BNB"
# combine the two
twenty_pnt10 <- rbind(twenty_a10, twenty_b10)
head(twenty_pnt10)


## ---- fig.width=8, fig.height=6--------------------------------------------------------------------------------------
# Long format
library(tidyr)
twenty_pnt10 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  # Plot line graphs
  ggplot(aes(y = value, x = ...1, group = model, col = model))+
  geom_line() +
  facet_wrap(~ metric, scales = "free") +
  # add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size n = 240, \u03B8 = 10",
       x = "Dataset number")


## --------------------------------------------------------------------------------------------------------------------
library(flextable)
s3_240 <- 
twenty_pnt10 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  group_by(metric,model) %>% 
  summarise(mean_val = mean(value)) %>% 
  spread(key = model, value = mean_val) %>% 
  mutate(n = 240, theta = 10)



s3_240 %>% 
  mutate_if(is.numeric,round, 2) %>% 
  regulartable()


## --------------------------------------------------------------------------------------------------------------------
# Read in the four time periods
thirty_a10 <- read_csv("./model_results/thirty_year_10_metrics.csv")
# add mdtype column
thirty_a10$model <- "NB"

thirty_b10 <- read_csv("./model_results/thirty_year_10b_metrics.csv")
# add mdtype
thirty_b10$model <- "BNB"
# combine the two
thirty_pnt10 <- rbind(thirty_a10, thirty_b10)
head(thirty_pnt10)


## ---- fig.width=8, fig.height=6--------------------------------------------------------------------------------------
# Long format
library(tidyr)
thirty_pnt10 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  # Plot line graphs
  ggplot(aes(y = value, x = ...1, group = model, col = model))+
  geom_line() +
  facet_wrap(~ metric, scales = "free") +
  # add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size n = 360, \u03B8 = 10",
       x = "Dataset number")


## --------------------------------------------------------------------------------------------------------------------
library(flextable)

s3_360 <- 
thirty_pnt10 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  group_by(metric,model) %>% 
  summarise(mean_val = mean(value)) %>% 
  spread(key = model, value = mean_val)%>% 
  mutate(n = 360, theta = 10)



s3_360 %>% 
  mutate_if(is.numeric,round, 2) %>% 
  regulartable()


## --------------------------------------------------------------------------------------------------------------------
# Read in the four time periods
five_a100 <- read_csv("./model_results/five_year_100_metrics.csv")
# add mdtype column
five_a100$model <- "NB"

five_b100 <- read_csv("./model_results/five_year_100b_metrics.csv")
# add mdtype
five_b100$model <- "BNB"
# combine the two
five_pnt100 <- rbind(five_a100, five_b100)
head(five_pnt100)


## ---- fig.width=8, fig.height=6--------------------------------------------------------------------------------------
# Long format
library(tidyr)
five_pnt100 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  # Plot line graphs
  ggplot(aes(y = value, x = ...1, group = model, col = model))+
  geom_line() +
  facet_wrap(~ metric, scales = "free") +
  # add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size n = 60, \u03B8 = 100",
       x = "Dataset number")


## --------------------------------------------------------------------------------------------------------------------
library(flextable)

s4_60 <- 
five_pnt100 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  group_by(metric,model) %>% 
  summarise(mean_val = mean(value)) %>% 
  spread(key = model, value = mean_val) %>% 
  mutate(n = 60, theta = 100)


s4_60 %>% 
  mutate_if(is.numeric,round, 2) %>% 
  regulartable()


## --------------------------------------------------------------------------------------------------------------------
# Read in the four time periods
ten_a100 <- read_csv("./model_results/ten_year_100_metrics.csv")
# add mdtype column
ten_a100$model <- "NB"

ten_b100 <- read_csv("./model_results/ten_year_100b_metrics.csv")
# add mdtype
ten_b100$model <- "BNB"
# combine the two
ten_pnt100 <- rbind(ten_a100, ten_b100)
head(ten_pnt100)


## ---- fig.width=8, fig.height=6--------------------------------------------------------------------------------------
# Long format
library(tidyr)
ten_pnt100 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  # Plot line graphs
  ggplot(aes(y = value, x = ...1, group = model, col = model))+
  geom_line() +
  facet_wrap(~ metric, scales = "free") +
  # add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size n = 120, \u03B8 = 100",
       x = "Dataset number")


## --------------------------------------------------------------------------------------------------------------------
library(flextable)
s4_120 <- 
ten_pnt100 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  group_by(metric,model) %>% 
  summarise(mean_val = mean(value)) %>% 
  spread(key = model, value = mean_val) %>% 
  mutate(n = 120, theta = 100)


s4_120 %>%
  mutate_if(is.numeric,round, 2) %>% 
  regulartable()


## --------------------------------------------------------------------------------------------------------------------
# Read in the four time periods
twenty_a100 <- read_csv("./model_results/twenty_year_100_metrics.csv")
# add mdtype column
twenty_a100$model <- "NB"

twenty_b100 <- read_csv("./model_results/twenty_year_100b_metrics.csv")
# add mdtype
twenty_b100$model <- "BNB"
# combine the two
twenty_pnt100 <- rbind(twenty_a100, twenty_b100)
head(five_pnt100)


## ---- fig.width=8, fig.height=6--------------------------------------------------------------------------------------
# Long format
library(tidyr)
twenty_pnt10 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  # Plot line graphs
  ggplot(aes(y = value, x = ...1, group = model, col = model))+
  geom_line() +
  facet_wrap(~ metric, scales = "free") +
  # add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size n = 240, \u03B8 = 100",
       x = "Dataset number")


## --------------------------------------------------------------------------------------------------------------------
library(flextable)
s4_240 <- 
twenty_pnt10 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  group_by(metric,model) %>% 
  summarise(mean_val = mean(value)) %>% 
  spread(key = model, value = mean_val)%>% 
  mutate(n = 240, theta = 100)


s4_240 %>% 
  mutate_if(is.numeric,round, 2) %>% 
  regulartable()


## --------------------------------------------------------------------------------------------------------------------
# Read in the four time periods
thirty_a100 <- read_csv("./model_results/thirty_year_100_metrics.csv")
# add mdtype column
thirty_a100$model <- "NB"

thirty_b100 <- read_csv("./model_results/thirty_year_100b_metrics.csv")
# add mdtype
thirty_b100$model <- "BNB"
# combine the two
thirty_pnt100 <- rbind(thirty_a100, thirty_b100)
head(thirty_pnt100)


## ---- fig.width=8, fig.height=6--------------------------------------------------------------------------------------
# Long format
library(tidyr)
thirty_pnt100 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  # Plot line graphs
  ggplot(aes(y = value, x = ...1, group = model, col = model))+
  geom_line() +
  facet_wrap(~ metric, scales = "free") +
  # add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size n = 360, \u03B8 = 100",
       x = "Dataset number")


## --------------------------------------------------------------------------------------------------------------------
library(flextable)
s4_360 <- 
thirty_pnt100 %>% 
  pivot_longer(cols = rmse_train:bias_test, names_to = "metric",
               values_to = "value") %>% 
  group_by(metric,model) %>% 
  summarise(mean_val = mean(value)) %>% 
  spread(key = model, value = mean_val)%>% 
  mutate(n = 360, theta = 100)


s4_360 %>% 
  mutate_if(is.numeric,round, 2) %>% 
  regulartable()


## --------------------------------------------------------------------------------------------------------------------
# combine data
final_sim <- rbind(s1_60, s1_120, s1_240, s1_360,
                   s2_60, s2_120, s2_240, s2_360,
                   s3_60, s3_120, s3_240, s3_360,
                   s4_60, s4_120, s4_240, s4_360)

# write out file for future
write.csv(final_sim, "paper_sim_results_pap.csv")


## --------------------------------------------------------------------------------------------------------------------
# Plot
final_sim %>% 
  filter(n == 60) %>% 
  pivot_longer(BNB:NB, names_to = "model", values_to = "value") %>% 
  mutate(label = ifelse(metric == "bias_test", paste("(a)","       Bias on test sets"),
                                                     ifelse(metric == "mase_test", paste("(b)",
                                                                                         "       MASE on test sets"),
                                                            ifelse(metric == "rmse_test",
                                                                   paste("(c)","       RMSE on test sets"), paste("(d)", "       RMSE on training sets"))))) %>% 
  ggplot(aes(x = theta, y = value, group = model, col = model)) +
  geom_line() +
  facet_wrap(~label, scales = "free")+# add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size = 60",
       x = "Dispersion parameter \u03B8")




## --------------------------------------------------------------------------------------------------------------------
# Plot
final_sim %>% 
  filter(n == 120) %>% 
  pivot_longer(BNB:NB, names_to = "model", values_to = "value") %>% 
  mutate(label = ifelse(metric == "bias_test", paste("(a)","       Bias on test sets"),
                                                     ifelse(metric == "mase_test", paste("(b)",
                                                                                         "       MASE on test sets"),
                                                            ifelse(metric == "rmse_test",
                                                                   paste("(c)","       RMSE on test sets"), paste("(d)", "       RMSE on training sets"))))) %>% 
  ggplot(aes(x = theta, y = value, group = model, col = model)) +
  geom_line() +
  facet_wrap(~label, scales = "free")+# add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size = 120",
       x = "Dispersion parameter \u03B8")


## --------------------------------------------------------------------------------------------------------------------
# Plot
final_sim %>% 
  filter(n == 240) %>% 
  pivot_longer(BNB:NB, names_to = "model", values_to = "value") %>% 
  mutate(label = ifelse(metric == "bias_test", paste("(a)","       Bias on test sets"),
                                                     ifelse(metric == "mase_test", paste("(b)",
                                                                                         "       MASE on test sets"),
                                                            ifelse(metric == "rmse_test",
                                                                   paste("(c)","       RMSE on test sets"), paste("(d)", "       RMSE on training sets"))))) %>% 
  ggplot(aes(x = theta, y = value, group = model, col = model)) +
  geom_line() +
  facet_wrap(~label, scales = "free")+# add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size = 240",
       x = "Dispersion parameter \u03B8")


## --------------------------------------------------------------------------------------------------------------------
# Plot
final_sim %>% 
  filter(n == 360) %>% 
  pivot_longer(BNB:NB, names_to = "model", values_to = "value") %>% 
  mutate(label = ifelse(metric == "bias_test", paste("(a)","       Bias on test sets"),
                                                     ifelse(metric == "mase_test", paste("(b)",
                                                                                         "       MASE on test sets"),
                                                            ifelse(metric == "rmse_test",
                                                                   paste("(c)","       RMSE on test sets"), paste("(d)", "       RMSE on training sets"))))) %>% 
  ggplot(aes(x = theta, y = value, group = model, col = model)) +
  geom_line() +
  facet_wrap(~label, scales = "free")+# add theme
  theme_bw()+
  # add labels
  labs(title = "Comparing the Bayesian Negative Binomial model and the Standard Negative Binomial model",
       subtitle = "Comparison of four metrics with sample size = 360",
       x = "Dispersion parameter \u03B8")


## --------------------------------------------------------------------------------------------------------------------
# Negbinner
negbinner2 <- function(x, prop = 0.8){
  
  # create training and test sets
  # set seed
  set.seed(456)
  
  n = length(x$count)
  trainIndex <- round(prop*n)
  # Create the data sets
  fireTrain <- x[1:trainIndex,]
  fireTest  <- x[(trainIndex+1):n,]
  
  # Fit model on training set
  
  glmNB <- MASS::glm.nb(count ~ mean_max_temp +
                           mean_rainfall +
                          #tyme +
                          sin((2*12*pi/tyme) +rnorm(1,sd=0.1))+
                       cos((2*12*pi/tyme) + rnorm(1,sd=0.1)),
                        data = fireTrain,
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
  test_bias <- Metrics::percent_bias(actual = fireTest$count,
                             predicted = round(predictions_test))
  # Dispersion parameter
  
  df = cbind(rmse_train = train_rmse, rmse_test = test_rmse, mase_test = test_mase,
        bias_test = test_bias, n = n, prop = prop)
  
  df
  
}


## --------------------------------------------------------------------------------------------------------------------
stanbinner3 <- function(x, prop = 0.8){
  
  # create training and test sets
  # set seed
  set.seed(456)
  n = length(x$count)
  x$tyme <- 1:n
  x$pi <- pi
  # add sine cosine
  x$tcos <- sin(2*pi/x$tyme)+ cos(2*pi/x$tyme)
  
  trainIndex <- round(prop*n)
  # Create the datasets
  fireTrain <- x[1:trainIndex,]
  fireTest  <- x[(trainIndex+1):n,]
  
  
  # Fit model on training set # Get priors
  
  brmNB <- rstanarm::stan_glm.nb(count ~  mean_max_temp +
                                    mean_rainfall +
                       #tcos,
                     # tyme +
                       sin((2*12*pi/tyme) +rnorm(1,sd=0.1))+
                       cos((2*12*pi/tyme) + rnorm(1,sd=0.1)),
                     iter = 2000,
                      data = fireTrain)
  
  
  
  # library(brms)
  predictions_train <- predict(brmNB,
   #                           newdata = fireTrain, type = "response")
  #predictions_train <- posterior_predict(brmNB,
                              newdata = fireTrain, type = "response")
  # Predict using train data
  predictions_train2 <- round(predictions_train)
  #predictions_train2 <- round(rowMeans(t(predictions_train)))
  #predictions_train2 <- round(t(predictions_train)[,1])
  # Predict on testing set
  predictions_test <- predict(brmNB,
                               newdata = fireTest, type = "response")
  #predictions_test <- posterior_predict(brmNB,
   #                           newdata = fireTest, type = "response")
  # Predict using test data
  predictions_test2 <- round(predictions_test)
  #predictions_test2 <- round(rowMeans(t(predictions_test)))
  #predictions_test2 <- round(t(predictions_test)[,1])
  # get the rmse
  train_rmse <- caret::RMSE(predictions_train2,fireTrain$count)
  
  test_rmse <- caret::RMSE(predictions_test2,fireTest$count)
  
  # get the MASE
  test_mase <- Metrics::mase(actual = fireTest$count,
                             predicted = predictions_test2)
  # Get the bias
  test_bias <- Metrics::percent_bias(actual = fireTest$count,
                                     predicted = predictions_test2)
  # Dispersion parameter
  
  df = cbind(rmse_train = train_rmse, rmse_test = test_rmse, mase_test = test_mase,
             bias_test = test_bias, n = n, prop = prop)
  
  df
}


## ---- message=FALSE--------------------------------------------------------------------------------------------------
# Script to fit models on data
library(MASS)
#library(stsm)
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
# read in the data
series_data <- read_csv("fire_data_2000-18.csv")
# add quarter
series_data2 <- series_data %>% 
  mutate(qrt = ifelse(month < 4, 1, ifelse(month > 3 & month < 7, 2,
                                                    ifelse(month > 9,
                                                           4, 3)))) %>% 
  mutate(tyme = 1:length(series_data$year)) 
  
  
  


## ---- message=FALSE--------------------------------------------------------------------------------------------------
# Set seed
set.seed(76568)

# Standard NB
nb_result80 <- negbinner2(series_data2, prop = 0.8)
nb_result90 <- negbinner2(series_data2, prop = 0.9)
nb_result95 <- negbinner2(series_data2, prop = 0.95)



# Bayesian NB
#bnb_result80 <- stanbinner2(series_data2, prop = 0.8)
#bnb_result90 <- stanbinner2(series_data2, prop = 0.9)
#bnb_result95 <- stanbinner2(series_data2, prop = 0.95)

# Bayesian NBT
bnbt_result80 <- stanbinner3(series_data2, prop = 0.8)
bnbt_result90 <- stanbinner3(series_data2, prop = 0.9)
bnbt_result95 <- stanbinner3(series_data2, prop = 0.95)


## --------------------------------------------------------------------------------------------------------------------
rbind(data.frame(nb_result80),
      data.frame(bnbt_result80)) %>% 
  data.frame() %>% 
  mutate_if(is.numeric, round, 2)

##
#-----------------------------------------------------------------------
# Get table to present on beta estimates
x=series_data2
set.seed(456)
n = length(x$count)
x$tyme <- 1:n
x$pi <- pi
# add sine cosine
x$tcos <- sin(2*pi/x$tyme)+ cos(2*pi/x$tyme)

trainIndex <- round(prop*n)
# Create the datasets
fireTrain <- x[1:trainIndex,]
fireTest  <- x[(trainIndex+1):n,]

# fit the model on whole dataset

brmNB <- rstanarm::stan_glm.nb(count ~  mean_max_temp +
                                 mean_rainfall +
                                 #tcos,
                                 # tyme +
                                 sin((2*12*pi/tyme) +rnorm(1,sd=0.1))+
                                 cos((2*12*pi/tyme) + rnorm(1,sd=0.1)),
                               iter = 2000,
                               data = x)
# Extract and tidy the output
tidied_output <- broom.mixed::tidy(brmNB)
# View the table
print(tidied_output)

# Get 95% credible intervals
credible_intervals <- posterior_interval(brmNB, prob = 0.95)
print(credible_intervals)

# combine table output
comb_table <- cbind(tidied_output,credible_intervals[-6,])
# remove rownames
rownames(comb_table) <- NULL

# View table
comb_table


## --------------------------------------------------------------------------------------------------------------------

# time series of rainfall
rain_ts = ts(series_data$mean_rainfall, frequency = 12, end = c(2018,12))

# time series of max temp
temp_ts = ts(series_data$mean_max_temp, frequency = 12, end = c(2018,12))

# time series of fire freq
fire_ts = ts(series_data$count, frequency = 12, end = c(2018,12))

# multivariate timeseries
rainfire_ts <- ts(series_data[,c("mean_rainfall","count")],
                  frequency = 12, end = c(2018,12))


## --------------------------------------------------------------------------------------------------------------------
# Decompose the monthly rain time series
library(forecast)
decomp <- stl(rain_ts, s.window = "periodic")
# plot decomposed series
autoplot(decomp) +
  labs(title = "Decomposed time series of monthly rainfall in Kenya (2000-2018)" )+
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.y = element_text(size = 14),
        axis.title.x = element_text(size = 14),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 13))


## --------------------------------------------------------------------------------------------------------------------
# decompose temperature time series
decomp2 <- stl(temp_ts, s.window = "periodic")
# plot decomposed series
autoplot(decomp2) +
  labs(title = "Decomposed time series of monthly maximum temperature in Kenya (2000-2018)" )+
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.y = element_text(size = 14),
        axis.title.x = element_text(size = 14),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 13))


## --------------------------------------------------------------------------------------------------------------------
# decompose fire count time series
decomp2 <- stl(fire_ts, s.window = "periodic")
# plot decomposed series
autoplot(decomp2) +
  labs(title = "Decomposed time series of monthly fire count in Kenya (2000-2018)" )+
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.title.y = element_text(size = 14),
        axis.title.x = element_text(size = 14),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 13))


## --------------------------------------------------------------------------------------------------------------------
# rainfall and fire frequency
corplot1 <- 
series_data %>% 
  mutate(month2 = factor(month, levels = 1:12,
                        labels = c("Jan","Feb","Mar","Apr","May","Jun",
                                   "Jul", "Aug","Sep", "Oct","Nov", "Dec"))) %>% 
  ggplot(aes(x = mean_rainfall, y = log(count)))+
  # add points
  geom_point() +
  # add regression line
  geom_smooth(formula = 'y~x', method = 'lm')+
  theme_bw()+
  labs(title = "Fire frequency vs rainfall",
       x = "Rainfall (mm)",
       y =  "logarithm of fire frequency") +
  facet_wrap( ~month2, scales = "free")+
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title.y = element_text(size = 16),
        axis.title.x = element_text(size = 16),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        strip.text = element_text(size = 16),
        legend.position = "none")
corplot1


## --------------------------------------------------------------------------------------------------------------------
# max temp and fire frequency
corplot2 <- 
series_data %>% 
  mutate(month2 = factor(month, levels = 1:12,
                        labels = c("Jan","Feb","Mar","Apr","May","Jun",
                                   "Jul", "Aug","Sep", "Oct","Nov", "Dec"))) %>% 
  ggplot(aes(x = mean_rainfall, y = log(count)))+
  # add points
  geom_point() +
  # add regression line
  geom_smooth(formula = 'y~x', method = 'lm')+
  theme_bw()+
  labs(title = "Fire frequency vs maximum temperature ",
       x = "Maximum temperature (\u00B0C)",
       y =  "logarithm of fire frequency") +
  facet_wrap( ~month2, scales = "free")+
  theme(plot.title = element_text(hjust = 0.5, size = 16),
        axis.title.y = element_text(size = 16),
        axis.title.x = element_text(size = 16),
        axis.text.x = element_text(size = 16),
        axis.text.y = element_text(size = 16),
        strip.text = element_text(size = 16),
        legend.position = "none")
corplot2


## --------------------------------------------------------------------------------------------------------------------
# average temp vs average fire count over time
temp_fire_avg = 
series_data2 %>% 
  mutate(era = ifelse(year < 2010, "00's", "10's")) %>% 
  group_by(year,era) %>% 
  summarize(mean_temp = mean(mean_max_temp),
            mean_count = mean(count)) %>% 
 # filter(year > 2000) %>% 
  ggplot(aes(x = mean_temp, y = log(mean_count),
             fill = era, col = era))+
  geom_point()+
  geom_smooth(formula = 'y~ x', method = "lm")+
  # add labels
  labs(title = "Relationship between mean maximum temperature and fire frequency",
       y = "log of fire frequency",
       x= "Maximum temperature (\u00B0C)") +
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5, size = 16),
         axis.title.y = element_text(size = 16, face = "bold"),
         axis.title.x = element_text(size = 16, face = "bold"),
         axis.text.x = element_text(size = 16),
         axis.text.y = element_text(size = 16),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16))
temp_fire_avg


## --------------------------------------------------------------------------------------------------------------------
# average rainfall vs average fire count over time
rain_fire_avg = 
series_data2 %>% 
  mutate(era = ifelse(year < 2010, "00's", "10's")) %>% 
  group_by(year,era) %>% 
  summarize(mean_rain = mean(mean_rainfall),
            mean_count = mean(count)) %>% 
 # filter(year > 2000) %>% 
  ggplot(aes(x = mean_rain, y = log(mean_count),
             fill = era, col = era))+
  geom_point()+
  geom_smooth(formula = 'y~ x', method = "lm")+
  # add labels
  labs(title = "Relationship between mean rainfall and fire frequency",
       y = "log of fire frequency",
       x = "Rainfall (mm)") +
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5, size = 16),
         axis.title.y = element_text(size = 16, face = "bold"),
         axis.title.x = element_text(size = 16, face = "bold"),
         axis.text.x = element_text(size = 16),
         axis.text.y = element_text(size = 16),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16))
rain_fire_avg


## --------------------------------------------------------------------------------------------------------------------
rbind(bnb_result80,bnb_result90, bnb_result95) %>% 
  data.frame()


## --------------------------------------------------------------------------------------------------------------------
# Fit model
n = length(series_data2$count)
trainIndex <- round(prop*n)
  # Create the data sets
fireTrain <- series_data2[1:trainIndex,]
fireTest  <- series_data2[(trainIndex+1):n,]
  
# Fit model on training set (NB)
  
glmNB <- MASS::glm.nb(count ~ mean_max_temp +
                           mean_rainfall +
                          sin((2*12*pi/tyme) +rnorm(1,sd=0.1))+
                                   cos((2*12*pi/tyme) + rnorm(1,sd=0.1)),
                        data = fireTrain,
                        link = "log")
# Fit Bayesian Model
stanNB <- rstanarm::stan_glm.nb(count ~  mean_max_temp +
                           mean_rainfall +
                             sin((2*12*pi/tyme) +rnorm(1,sd=0.1))+
                                   cos((2*12*pi/tyme) + rnorm(1,sd=0.1)),
                                 data = fireTrain,
                                 link = "log")


## --------------------------------------------------------------------------------------------------------------------
# prediction intervals NB
library(ciTools)
intervals_nb <- add_pi(df = fireTest, fit = glmNB, names = c("lower", "upper"))
# Write
write.csv(intervals_nb, "prediction_intervals2.csv")
intervals_nb



## --------------------------------------------------------------------------------------------------------------------
# prediction intervals BNB
intervals_bnb <- predictive_interval(stanNB, newdata = fireTest, prob = 0.95)
# Predict using test data
prd2 = rstanarm::posterior_predict(stanNB, newdata = fireTest)
predictions_test2 <- t(prd2)[,1]
# add test data
preds_intervals <- cbind(intervals_bnb, actual = fireTest$count, predictions = predictions_test2)
# add col names
colnames(preds_intervals) <- c("lower", "upper", "count","pred")
# Write
write.csv(preds_intervals, "prediction_intervals_pap.csv")
preds_intervals


## --------------------------------------------------------------------------------------------------------------------
# Review prediction intervals
pred_intBNB <- read.csv("prediction_intervals_pap.csv")
# Rename variables
head(pred_intBNB)


## --------------------------------------------------------------------------------------------------------------------
# combine the two datasets
# NB
# read in intervals
pred_intNB <- read.csv("prediction_intervals2.csv")
nbdata <- pred_intNB[,c("count","lower","upper","pred")]
# round preds
nbdata$pred = round(nbdata$pred)
# add model name
nbdata$model <- "NB"

# BNB
bnbdata <- pred_intBNB[,c("count","lower","upper","pred")]
# add model name
bnbdata$model <- "BNB"

# Merge data
pidata <- rbind(nbdata, bnbdata)

# Add time component
timer= series_data2[175:218,]
timer$month_name <- month.abb[timer$month]
timer$month_year <- paste0(timer$month_name, "-", timer$year )
# change to date
timer$month_year2 <- zoo::as.yearmon(timer$month_year,
                                     "%b-%Y")

# Add to pie data
pidata$month_year <- rep(timer$month_year2, 2)



## --------------------------------------------------------------------------------------------------------------------
# Create data frame
df <- data.frame(x = 1:44,
                 y = nbdata$count,
                 modelNB_pred = nbdata$pred,
                 modelNB_upper = nbdata$upper,
                 modelNB_lower = nbdata$lower,
                 modelBNB_pred = bnbdata$pred,
                 modelBNB_upper = bnbdata$upper,
                 modelBNB_lower = bnbdata$lower)



ggplot(df, aes(x = x)) +
  geom_line(aes(y = y), color = "black", linewidth = 1) +
  geom_ribbon(aes(ymin = modelNB_lower, ymax = modelNB_upper), 
              fill = "blue", alpha = 0.3) +
  geom_line(aes(y = modelNB_pred), color = "blue", linewidth = 1) +
  labs(x = "Data Points", y = "Fire frequency") +
  theme_bw()

ggplot(df, aes(x = x)) +
geom_line(aes(y = y), color = "black", linewidth = 1) +
  geom_ribbon(aes(ymin = modelBNB_lower, ymax = modelBNB_upper), 
              fill = "red", alpha = 0.3) +
  geom_line(aes(y = modelBNB_pred), color = "red", linewidth = 1) +
  labs(x = "Data Points", y = "Fire frequency") +
  theme_bw()

# Included plot

ggplot(pidata, aes(x = month_year)) +
geom_line(aes(y = count), color = "black", linewidth = 1,
          linetype = 1) +
  geom_ribbon(aes(ymin = lower, ymax = upper), 
              fill = "blue", alpha = 0.2) +
  geom_line(aes(y = pred), color = "red", linewidth = .5) +
  labs(x = "Year of Observation", y = "Fire frequency",
  title = "Comparison of predictions and prediction intervals for the models",
  subtitle = "Shaded area is the predictive interval, red thin line \nis the predicted values and black line is the actual values") +
  theme_bw() +
  facet_wrap(~model)+
  theme(plot.title = element_text(hjust = 0.5, size = 20),
                                  axis.text = element_text( size = 14 ),
        plot.subtitle = element_text(hjust = 0.5, size = 16),
           axis.text.x = element_text( size = 16),
        axis.text.y = element_text( size = 16),
           axis.title = element_text( size = 16, face = "bold" ),
           legend.position="none",
           # The new stuff
           strip.text = element_text(size = 20),
        # Add white space on the right side of the plot
    plot.margin = margin(t = 20, r = 50, b = 20, l = 20, unit = "pt")
  )



## --------------------------------------------------------------------------------------------------------------------
# check relationship between observed and predicted values.
# bnb
cor.test(bnbdata$count, bnbdata$pred, pa)
# nb
cor.test(nbdata$count, nbdata$pred)


## --------------------------------------------------------------------------------------------------------------------
# plot a prediction interval curve
library(tidyr)
nbmodel <- 
nbdata %>% 
  rename(actual = count) %>% 
  pivot_longer(lower:pred, names_to = "interval",
               values_to = "count") %>% 
  arrange(interval) %>% 
  mutate(index = rep(1:44,3)) %>% 
  ggplot(aes(x = index, y = count, group = interval,
             col = interval)) +
  geom_line() +
  labs(title = "Prediction intervals of the NB model compared to the predictions",
       subtitle = "Values obtained from the predictions on the testing dataset at 0.95 desired probability", x = "Index of value")+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        axis.title.y = element_text(size = 14),
        axis.title.x = element_text(size = 14),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
           # The new stuff
           strip.text = element_text(size = 14))


## --------------------------------------------------------------------------------------------------------------------
# Plot the figures
bnbmodel <- 
bnbdata %>% data.frame() %>% 
  rename(actual = count) %>% 
  pivot_longer(lower:pred, names_to = "interval",
               values_to = "count") %>% 
  arrange(interval) %>% 
  mutate(index = rep(1:44,3)) %>% 
  ggplot(aes(x = index, y = count, group = interval,
             col = interval)) +
  geom_line() +
  labs(title = "Prediction intervals of the BNB model compared to the predictions",
       subtitle = "Values obtained from the predictions on the testing dataset at 0.95 desired probability", x = "Index of value")+
  theme_bw()+
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        axis.title.y = element_text(size = 14),
        axis.title.x = element_text(size = 14),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14),
           # The new stuff
           strip.text = element_text(size = 14))


## --------------------------------------------------------------------------------------------------------------------
# combine diagram
# Combine the plots
library(patchwork)
combinepi <- (nbmodel / bnbmodel)
combinepi

