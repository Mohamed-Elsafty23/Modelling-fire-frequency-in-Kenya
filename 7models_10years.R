# Script to run models on 10 year data
# Source model formulas

source('final_models.R')
library(data.table)

# Set seed
set.seed(76568)

# Run standard Negative Binomial model
#########################################################################



################################################################
# 10 - YEARS
# Results for 10 year - 1.5
ten_year1.5 <- list.files(path = "./simulated_data/d10year/theta_1.5",
                          # Identify all csv files in folder
                          pattern = "*.csv", full.names = TRUE) %>% 
  lapply(fread) %>%             # Store all files in list
  map(negbinner) 
ten_year1.5 <- do.call(rbind, ten_year1.5)        # Combine data sets into one data set 
# check the data
head(ten_year1.5)  
# write.csv
write.csv(ten_year1.5, "./model_results/ten_year_1.5_metrics.csv")
#______________________________________________________________________

# Results for 10 year - 5
ten_year5 <- list.files(path = "./simulated_data/d10year/theta_5",
                        # Identify all csv files in folder
                        pattern = "*.csv", full.names = TRUE) %>% 
  lapply(fread) %>%             # Store all files in list
  map(negbinner) 
ten_year5 <- do.call(rbind, ten_year5)        # Combine data sets into one data set 
# check the data
head(ten_year5)  
# write.csv
write.csv(ten_year5, "./model_results/ten_year_5_metrics.csv")

#______________________________________________________________________

# Results for 10 year - 10
ten_year10 <- list.files(path = "./simulated_data/d10year/theta_10",
                         # Identify all csv files in folder
                         pattern = "*.csv", full.names = TRUE) %>% 
  lapply(fread) %>%             # Store all files in list
  map(negbinner) 
ten_year10 <- do.call(rbind, ten_year10)        # Combine data sets into one data set 
# check the data
head(ten_year10)  
# write.csv
write.csv(ten_year10, "./model_results/ten_year_10_metrics.csv")

#______________________________________________________________________

# Results for 10 year - 100
ten_year100 <- list.files(path = "./simulated_data/d10year/theta_100",
                          # Identify all csv files in folder
                          pattern = "*.csv", full.names = TRUE) %>% 
  lapply(fread) %>%             # Store all files in list
  map(negbinner)
ten_year100 <- do.call(rbind, ten_year100)        # Combine data sets into one data set 
# check the data
head(ten_year100)  
# write.csv
write.csv(ten_year100, "./model_results/ten_year_100_metrics.csv")

###############################################################


##### BAYESIAN MODEL

# Run Bayesian Negative Binomial model
#########################################################################
# Results for 10 year - 1.5
ten_year1.5b <- list.files(path = "./simulated_data/d10year/theta_1.5",
                            # Identify all csv files in folder
                            pattern = "*.csv", full.names = TRUE) %>% 
  # Use fread from data.table to read in the files and store them in a list
  lapply(fread) %>%  
  # Use parLapply from parallel to apply stanbinner to each dataset in parallel
  parLapply(cl = cl, fun = stanbinner) 
ten_year1.5b <- do.call(rbind, ten_year1.5b)        # Combine data sets into one data set 
# check the data
head(ten_year1.5b)  
# write.csv
write.csv(ten_year1.5b, "./model_results/ten_year_1.5b_metrics.csv")
#______________________________________________________________________

# Results for 10 year - 5
ten_year5b <- list.files(path = "./simulated_data/d10year/theta_5",
                          # Identify all csv files in folder
                          pattern = "*.csv", full.names = TRUE) %>% 
  # Use fread from data.table to read in the files and store them in a list
  lapply(fread) %>%  
  # Use parLapply from parallel to apply stanbinner to each dataset in parallel
  parLapply(cl = cl, fun = stanbinner) 
ten_year5b <- do.call(rbind, ten_year5b)        # Combine data sets into one data set 
# check the data
head(ten_year5b)  
# write.csv
write.csv(ten_year5b, "./model_results/ten_year_5b_metrics.csv")

#______________________________________________________________________

# Results for 10 year - 10
ten_year10b <- list.files(path = "./simulated_data/d10year/theta_10",
                           # Identify all csv files in folder
                           pattern = "*.csv", full.names = TRUE) %>% 
  # Use fread from data.table to read in the files and store them in a list
  lapply(fread) %>%  
  # Use parLapply from parallel to apply stanbinner to each dataset in parallel
  parLapply(cl = cl, fun = stanbinner) 
ten_year10b <- do.call(rbind, ten_year10b)        # Combine data sets into one data set 
# check the data
head(ten_year10b)  
# write.csv
write.csv(ten_year10b, "./model_results/ten_year_10b_metrics.csv")

#______________________________________________________________________

# Results for 10 year - 100
ten_year100b <- list.files(path = "./simulated_data/d10year/theta_100",
                            # Identify all csv files in folder
                            pattern = "*.csv", full.names = TRUE) %>% 
  # Use fread from data.table to read in the files and store them in a list
  lapply(fread) %>%  
  # Use parLapply from parallel to apply stanbinner to each dataset in parallel
  parLapply(cl = cl, fun = stanbinner) 
ten_year100b <- do.call(rbind, ten_year100b)        # Combine data sets into one data set 
# check the data
head(ten_year100b)  
# write.csv
write.csv(ten_year100b, "./model_results_pap/ten_year_100b_metrics.csv")
