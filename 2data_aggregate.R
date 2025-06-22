
# Script to create dataset to be used for analyses.
library(tidyverse)

# Read in all finalized fire, temperature and rainfall datasets

fire_all <- list.files(path = "./climate",     # Identify all csv files in folder
                        pattern = "*.csv", full.names = TRUE) %>% 
  lapply(read_csv) %>%             # Store all files in list
  bind_rows()          # Combine data sets into one data set 
# check the data
head(fire_all)  

# Add year and month column
library(lubridate)

# Group by month and year
fire_data <- 
  fire_all %>% 
  # group by 
  group_by(month, year) %>% 
  # summarize
  summarize(count = n(), mean_brightness = mean(brightness, na.rm = TRUE),
            mean_bright31 = mean(bright_t31, na.rm = T),
            mean_frp = mean(frp, na.rm = T),
            mean_max_temp = mean(max_temp, na.rm = T),
            mean_min_temp = mean(min_temp, na.rm = T),
            mean_rainfall = mean(rainfall, na.rm = T),
            anomaly = mean_max_temp - mean_min_temp,
            average_temp = mean(mean_min_temp,mean_max_temp)) %>% 
  # arrange 
  arrange(year)

# Write final dataset
write.csv(fire_data,"fire_data_2000-18.csv")


