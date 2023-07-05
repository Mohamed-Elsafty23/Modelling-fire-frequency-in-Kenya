# Import and merge files
library(tidyverse)
# List and import all MODIS files on fire frequency in Kenya
# Data runs from November 2000 to Decemeber 2020
modis_all <- list.files(path = "./modis_data",     # Identify all csv files in folder
                       pattern = "*.csv", full.names = TRUE) %>% 
  lapply(read_csv) %>%        # Store all files in list
  bind_rows()              # Combine data sets into one data set 
# check the data
head(modis_all)  

# Add year and month column
library(lubridate)
modis_all <- 
  modis_all %>% 
  # add year and month
  mutate(year = year(acq_date),
         month = month(acq_date))



# Importing maximum temperature files to extract temperature data
library(raster)
library(stars)
library(sf)
library(sp)
library(stringr)

# Create fire data files
modis_all <- modis_all %>% group_by(year,month)

# Set the path for the new folder
folder_name <- "fire"

# Create the folder in the working directory
dir.create(folder_name)

# Function to write csv files for each year month combination
customFun  = function(DF) {
  write.csv(DF,paste0("fire/","fire_year_",unique(DF$year),"-",
  unique(DF$month),".csv"))
  return(DF)
}

# write csv files
modis_all %>% 
  do(customFun(.))


# Read in list of tif files and csv and create sf object

fire_names <- list.files(path = "./fire",     # Identify all csv files in folder
                         pattern = "*.csv", full.names = TRUE)

tif_names_tmax <- list.files(path = "./tmax",     # Identify all tif files in folder
                          pattern = "*.tif", full.names = TRUE)

tif_names_tmin <- list.files(path = "./tmin",     # Identify all tif files in folder
                             pattern = "*.tif", full.names = TRUE)

tif_names_rain <- list.files(path = "./rain",     # Identify all tif files in folder
                             pattern = "*.tif", full.names = TRUE)



# Set the path for the new folder
folder_name2 <- "climate"

# Create the folder in the working directory
dir.create(folder_name2)



# Loop to intersect fire data with the climatic variables
  
for (i in 1:length(fire_names)){
    # read tmax tif file
    tmax_file <- st_as_sf(read_stars(tif_names_tmax[[i]]))
    # read fire file
    fire_file <- read.csv(fire_names[[i]])
    # read tmin tif file
    tmin_file <- st_as_sf(read_stars(tif_names_tmin[[i]]))
    # read rain tif file
    rain_file <- st_as_sf(read_stars(tif_names_rain[[i]]))
    
    # tmax
    # create sf object with fire data
    fire_sf <- st_as_sf(fire_file, coords = c("longitude","latitude"),
                                crs = st_crs(tmin_file)) 
    sf::sf_use_s2(FALSE)
    # Calculate intersection and extract max temperature 
    # Max temperature
    pnts_file_max <- fire_sf %>% 
      mutate(
        intersection = as.character(st_intersects(geometry, tmax_file))) %>% 
      mutate(intersection = ifelse(str_count(intersection) > 8,'',
                                   intersection)) %>% 
      mutate(intersection = as.integer(intersection)) %>% 
      
      mutate(max_temp = if_else(is.na(intersection),
                                '', as.character(
                                  tmax_file[[1]][intersection])))
      # Min temperature
    pnts_file_min <- fire_sf %>% 
      mutate(
        intersection = as.character(st_intersects(geometry, tmin_file))) %>%
      # tweak to bypass instances where the intersection is two polygons
      mutate(intersection = ifelse(str_count(intersection) > 8,'',
                                   intersection)) %>% 
      mutate(intersection = as.integer(intersection)) %>% 
      
      mutate(min_temp = if_else(is.na(intersection),
                           '', as.character(
                             tmin_file[[1]][intersection])))
      
      # Precipitation
    pnts_file_rain <- fire_sf %>% 
      mutate(
        intersection = as.character(st_intersects(geometry, rain_file))) %>% 
      mutate(intersection = ifelse(str_count(intersection) > 8,'',
                                   intersection)) %>% 
      mutate(intersection = as.integer(intersection)) %>% 
      
      mutate(rainfall = if_else(is.na(intersection),
                                '', as.character(
                                  rain_file[[1]][intersection])))
      
      pnts_file <- pnts_file_max %>% 
        mutate(min_temp = pnts_file_min$min_temp,
               rainfall = pnts_file_rain$rainfall) %>% 
      
      data.frame() %>% 
      # make temperature numeric and correct values that did not intersect
      mutate(max_temp = as.numeric(max_temp),
             min_temp = as.numeric(min_temp),
             rainfall = as.numeric(rainfall)) %>% 
      mutate(max_temp = ifelse(max_temp > 0, max_temp, NA),
             min_temp = ifelse(min_temp > 0, min_temp, NA),
             rainfall = ifelse(rainfall > 0, rainfall, NA))
    
    # Remove geometry column
    pnts_file2 <- pnts_file[,-c(17,18)]
      
    # Write csv file
    write.csv(pnts_file2,
              paste0("climate/fire-tmax_",
                     substr(tif_names_tmax[i],13,19),".csv"))
    
    }
  



 

