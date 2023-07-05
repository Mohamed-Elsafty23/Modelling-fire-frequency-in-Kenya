# Script to run models on 5 year data

# Source model formulas

source('final_models.R')
library(data.table)

# Set seed
set.seed(76568)

# Run standard Negative Binomial model
#########################################################################
# Results for 5 year - 1.5
five_year1.5 <- list.files(path = "./simulated_data/d5year/theta_1.5",
                           # Identify all csv files in folder
                           pattern = "*.csv", full.names = TRUE) %>% 
  mclapply(fread) %>%             # Store all files in list
  map(negbinner) 
five_year1.5 <- do.call(rbind, five_year1.5)        # Combine data sets into one data set 
# check the data
head(five_year1.5)  
# write.csv
write.csv(five_year1.5, "./model_results/five_year_1.5_metrics.csv")
beepr::beep(sound = 1)
#______________________________________________________________________

# Results for 5 year - 5
five_year5 <- list.files(path = "./simulated_data/d5year/theta_5",
                         # Identify all csv files in folder
                         pattern = "*.csv", full.names = TRUE) %>% 
  mclapply(fread) %>%             # Store all files in list
  map(negbinner) 
five_year5 <- do.call(rbind, five_year5)        # Combine data sets into one data set 
# check the data
head(five_year5)  
# write.csv
write.csv(five_year5, "./model_results/five_year_5_metrics.csv")

#______________________________________________________________________

# Results for 5 year - 10
five_year10 <- list.files(path = "./simulated_data/d5year/theta_10",
                          # Identify all csv files in folder
                          pattern = "*.csv", full.names = TRUE) %>% 
  mclapply(fread) %>%              # Store all files in list
  map(negbinner) 
five_year10 <- do.call(rbind, five_year10)        # Combine data sets into one data set 
# check the data
head(five_year10)  
# write.csv
write.csv(five_year10, "./model_results/five_year_10_metrics.csv")

#______________________________________________________________________

# Results for 5 year - 100
five_year100 <- list.files(path = "./simulated_data/d5year/theta_100",
                           # Identify all csv files in folder
                           pattern = "*.csv", full.names = TRUE) %>% 
  mclapply(fread) %>%            # Store all files in list
  map(negbinner) 
five_year100 <- do.call(rbind, five_year100)        # Combine data sets into one data set 
# check the data
head(five_year100)  
# write.csv
write.csv(five_year100, "./model_results/five_year_100_metrics.csv")##

beepr::beep(sound = 1)

##### BAYESIAN MODEL

# Run Bayesian Negative Binomial model
#########################################################################
# Results for 5 year - 1.5
source('final_models.R')
library(data.table)

### Replace code
# Load parallel package
library(parallel)
library(data.table)
library(rstanarm)
cl <- makeCluster(getOption("cl.cores", 8))

five_year1.5b <- list.files(path = "./simulated_data/d5year/theta_1.5",
                            # Identify all csv files in folder
                            pattern = "*.csv", full.names = TRUE)  %>% 
  # Use fread from data.table to read in the files and store them in a list
  lapply(fread) %>%  
  # Use parLapply from parallel to apply stanbinner to each dataset in parallel
  parLapply(cl = cl, fun = stanbinner) 
# Use rbindlist from data.table to combine the datasets into a single data set
five_year1.5b <- do.call(rbind, five_year1.5b)        
# check the data
head(five_year1.5b)  
# write.csv
write.csv(five_year1.5b, "./model_results/five_year_1.5b_metrics.csv")
beepr::beep(sound = 1)

#______________________________________________________________________

# Results for 5 year - 5
five_year5b <- list.files(path = "./simulated_data/d5year/theta_5",
                         # Identify all csv files in folder
                         pattern = "*.csv", full.names = TRUE) %>% 
  # Use fread from data.table to read in the files and store them in a list
  lapply(fread) %>%  
  # Use parLapply from parallel to apply stanbinner to each dataset in parallel
  parLapply(cl = cl, fun = stanbinner) 
five_year5b <- do.call(rbind, five_year5b)        # Combine data sets into one data set 
# check the data
head(five_year5b)  
# write.csv
write.csv(five_year5b, "./model_results/five_year_5b_metrics.csv")

#______________________________________________________________________

# Results for 5 year - 10
five_year10b <- list.files(path = "./simulated_data/d5year/theta_10",
                          # Identify all csv files in folder
                          pattern = "*.csv", full.names = TRUE) %>% 
  # Use fread from data.table to read in the files and store them in a list
  lapply(fread) %>%  
  # Use parLapply from parallel to apply stanbinner to each dataset in parallel
  parLapply(cl = cl, fun = stanbinner) 
five_year10b <- do.call(rbind, five_year10b)        # Combine data sets into one data set 
# check the data
head(five_year10b)  
# write.csv
write.csv(five_year10b, "./model_results/five_year_10b_metrics.csv")

#______________________________________________________________________

# Results for 5 year - 100
five_year100b <- list.files(path = "./simulated_data/d5year/theta_100",
                           # Identify all csv files in folder
                           pattern = "*.csv", full.names = TRUE) %>% 
  # Use fread from data.table to read in the files and store them in a list
  lapply(fread) %>%  
  # Use parLapply from parallel to apply stanbinner to each dataset in parallel
  parLapply(cl = cl, fun = stanbinner) 
five_year100b <- do.call(rbind, five_year100b)        # Combine data sets into one data set 
# check the data
head(five_year100b)  
# write.csv
write.csv(five_year100b, "./model_results/five_year_100b_metrics.csv")
