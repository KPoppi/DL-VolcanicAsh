# function taken from https://dachro.github.io/ogh_summer_school_2020/Tutorial_DL_UAV.html#from_single_images_to_maps
# author: Christian Knoth
dl_subsets <- function(inputrst, targetsize, targetdir, targetname, img_info_only = FALSE, is_mask = FALSE) {
  require(raster)
  
  # determine next number of quadrats in x and y direction, by simple rounding
  targetsizeX <- targetsize[1]
  targetsizeY <- targetsize[2]
  inputX <- ncol(inputrst)
  inputY <- nrow(inputrst)
  
  # determine dimensions of raster so that 
  # it can be split by whole number of subsets (by shrinking it)
  while(inputX%%targetsizeX!=0){
    inputX = inputX-1  
  }
  while(inputY%%targetsizeY!=0){
    inputY = inputY-1    
  }
  
  # determine difference
  diffX <- ncol(inputrst)-inputX
  diffY <- nrow(inputrst)-inputY
  
  # determine new dimensions of raster and crop, 
  # cutting evenly on all sides if possible
  newXmin <- floor(diffX/2)
  newXmax <- ncol(inputrst)-ceiling(diffX/2)-1
  newYmin <- floor(diffY/2)
  newYmax <- nrow(inputrst)-ceiling(diffY/2)-1
  rst_cropped <- suppressMessages(crop(inputrst, extent(inputrst,newYmin,newYmax,newXmin,newXmax)))
  #writeRaster(rst_cropped,filename = target_dir_crop)
  #return (list(ssizeX = ssizeX, ssizeY = ssizeY, nsx = nsx, nsy =nsy))
  agg <- suppressMessages(aggregate(rst_cropped[[1]],c(targetsizeX,targetsizeY)))
  agg[]    <- suppressMessages(1:ncell(agg))
  agg_poly <- suppressMessages(rasterToPolygons(agg))
  names(agg_poly) <- "polis"
  pb <- txtProgressBar(min = 0, max = ncell(agg), style = 3)
  for(i in 1:ncell(agg)) {
    # rasterOptions(tmpdir=tmpdir)
    setTxtProgressBar(pb, i)
    e1  <- extent(agg_poly[agg_poly$polis==i,])
    subs <- suppressMessages(crop(rst_cropped,e1))
    
    writeRaster(subs, filename=paste0(targetdir, targetname, i, ".tif"), overwrite=TRUE) 
    #return(c(extent(rst_cropped),crs(rst_cropped)))
  }
  close(pb)
  #img_info <- list("tiles_rows"=nrow(rst_cropped)/targetsizeY, "tiles_cols"=ncol(rst_cropped)/targetsizeX,"crs"= crs(rst_cropped),"extent"=extent(rst_cropped))
  #writeRaster(rst_cropped,filename = paste0(targetdir,"input_rst_cropped.tif"))
  rm(subs,agg,agg_poly)
  gc()
  return(rst_cropped)
}


# function taken from https://dachro.github.io/ogh_summer_school_2020/Tutorial_DL_UAV.html#from_single_images_to_maps
# author: Christian Knoth
rebuild_img <- function(pred_subsets, out_path, target_rst) {
  require(raster)
  require(gdalUtils)
  require(stars)
  
  subset_pixels_x <- ncol(pred_subsets[1,,,])
  subset_pixels_y <- nrow(pred_subsets[1,,,])
  tiles_rows <- nrow(target_rst)/subset_pixels_y
  tiles_cols <- ncol(target_rst)/subset_pixels_x
  
  # load target image to determine dimensions
  target_stars <- st_as_stars(target_rst,proxy=F)
  #prepare subfolder for output
  result_folder <- paste0(out_path,"out")
  if(dir.exists(result_folder)){
    unlink(result_folder,recursive = T)
  }
  dir.create(path = result_folder)
  
  #for each tile, create a stars from corresponding predictions, 
  #assign dimensions using original/target image, and save as tif: 
  for (crow in 1:tiles_rows){
    for (ccol in 1:tiles_cols){
      i <- (crow-1)*tiles_cols + (ccol-1) +1 
      dimx <- c(((ccol-1)*subset_pixels_x+1),(ccol*subset_pixels_x))
      dimy <- c(((crow-1)*subset_pixels_y+1),(crow*subset_pixels_y))
      cstars <- st_as_stars(t(pred_subsets[i,,,1]))
      attr(cstars,"dimensions")[[2]]$delta=-1
      #set dimensions using original raster
      st_dimensions(cstars) <- st_dimensions(target_stars[,dimx[1]:dimx[2],dimy[1]:dimy[2]])[1:2]
      write_stars(cstars,dsn = paste0(result_folder,"/_out_",i,".tif")) 
    }
  }
  starstiles <- as.vector(list.files(result_folder,full.names = T),mode = "character")
  gdalbuildvrt(starstiles,paste0(result_folder,"/mosaic.vrt"))
  gdalwarp(paste0(result_folder,"/mosaic.vrt"), paste0(result_folder,"/mosaic.tif"))
}
