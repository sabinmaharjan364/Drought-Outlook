ibrary(raster)
library(ncdf4)
ncname <- "C:/Users/u1003670/Downloads/cdi_1.nc" 
dname <- "cdi" 
# open a netCDF file
ncin <- nc_open(ncname)
# get longitude and latitude
lon <- ncvar_get(ncin,"longitude")
nlon <- dim(lon)
head(lon)
lat <- ncvar_get(ncin,"latitude")
nlat <- dim(lat)
# get cdi   
cdi_array <- ncvar_get(ncin,dname)
dim(cdi_array)
# get a single slice or layer
cdi_slice <- cdi_array[306, , ]
# vector of `cdi` values
cdi_vec <- as.vector(cdi_slice)
cdi_df <- cbind(expand.grid(lon=lon, lat=lat), cdi= cdi_vec)
 
ncname <- "C:/Users/u1003670/Downloads/2024.forecast.nc"
# open a netCDF file
ncin <- nc_open(ncname)
dname <- "percentage_of_ensembles"
# get longitude and latitude
lon <- ncvar_get(ncin,"lon")
nlon <- dim(lon)
lat <- ncvar_get(ncin,"lat")
nlat <- dim(lat)
# get rain
rain_array <- ncvar_get(ncin,dname)
dim(rain_array)
# get slices
rain_slice <- rain_array[ , ,1,2]
rain_vec <- as.vector(rain_slice)
rain_df <- cbind(expand.grid(lon=round(lon,2), lat=round(lat,2)), rain = rain_vec)
 
# join by lat/lon
library(dplyr)
join_df <- left_join(cdi_df, rain_df, by=c("lon", "lat"))
 
# loop for drought category
library(foreach)
library(parallel)
library(doParallel)
rmna_df <- na.omit(join_df)
ncell <- dim(rmna_df)[1]
ncores <- detectCores()
cl <- makeCluster(ncores - 4)
registerDoParallel(cl)
classified <- foreach(i=1:ncell, .combine=rbind) %dopar% {
    if (rmna_df$cdi[i] <= 0.2) {
      if (rmna_df$rain[i] < 50) {
        if (rmna_df$cdi[i] <= 0.02) {
          5 # persists
        } else {
          6 # worsens
        }
      } else if (rmna_df$rain[i] < 70) {
        5 # persists
      } else {
        if (rmna_df$cdi[i] > 0.1 & rmna_df$cdi[i] <= 0.2) {
          2 # removed
        } else {
          3 # improved
        }
      }
    } else {
    if (rmna_df$rain[i] < 30) {
      4 # develops
    } else {
      1 # no drought
    }
  }
}
stopCluster(cl)
# rmna_df$category <- as.integer(classified[,1])
# create a dataframe contain the result
df_out <- as.data.frame(cbind(lon = cdi_df$lon, lat = cdi_df$lat, category = NA))
# get the rows where NAs were removed
order <- as.integer(rownames(rmna_df))
# replace the category value
df_out$category[order] <- as.integer(classified[,1])
 
library(sf)
# create spatial points data frame
spg <- df_out
coordinates(spg) <- ~ lon + lat
# coerce to SpatialPixelsDataFrame
gridded(spg) <- TRUE
# coerce to raster
rasterDF <- raster(spg)
plot(rasterDF)