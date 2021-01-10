# Code for study project "Einsatz von Cloud Computing Services fuer Deep Learning in der Fernerkundung"
# in the winter semester 2020/2021 at the University of Muenster
# by Fabian Fermazin and Katharina Poppinga

library(reticulate)
library(tensorflow)
library(keras)
library(purrr)
library(rsample)
library(abind)
library(sf)
library(stars)
library(tfdatasets) # tf: tensorflow
library(digest)
library(ggplot2)
library(sp)
library(raster)
library(mapview)
library(gdalUtils)
library(magick)

#reticulate::install_miniconda()
#keras::install_keras()

reticulate::py_config() # "reticulate::" ersetzt "library(reticulate)"
tensorflow::tf_config()
keras::is_keras_available()

setwd("/home/sp_gruppe2/StudyProject/Volcanos") # path for AWS-instance
getwd()

source("handle_subsets.R")
source("tif_processing.R")
source("CNN_pixel-based.R")

# TODO use functions dl_subsets and rebuild_img
# TODO make paths more flexible for all volcanos

# *****************************************************************************************************


### make subset-tiles from original Sentinel-2 image by using function 'dl_subsets' from 'handle_subsets.R':

# TODO Pfad an AWS-Instanz anpassen
etna_sentinel_full <- stack("D:/KPopp/Documents/WWU_Muenster/Semester_7/Studienprojekt_DL_CCS/Projekt/Volcanos/etna_data/etna_b2_b3_b4_b8_b12.tif")

# TODO sakurajima_sentinel_full <- stack("D:/KPopp/Documents/WWU_Muenster/Semester_7/Studienprojekt_DL_CCS/Projekt/Volcanos/etna_data/sakurajima_b2_b3_b4_b8_b12.tif")
# TODO suwanosejima_sentinel_full <- stack("D:/KPopp/Documents/WWU_Muenster/Semester_7/Studienprojekt_DL_CCS/Projekt/Volcanos/etna_data/suwanosejima_b2_b3_b4_b8_b12.tif")

sentinel_subsets = dl_subsets(inputrst = etna_sentinel_full,
                              targetsize = c(448,448),
                              targetdir = "D:/KPopp/Documents/WWU_Muenster/Semester_7/Studienprojekt_DL_CCS/Projekt/Volcanos/etna_data/train_and_test/pixel-based/imgs/", # must already exist
                              targetname = "etna_subset_")


# TODO MASKENERSTELLUNG in QGIS


### data preprocessing with data augmentation:

# make data.frame with full paths of images and their masks
# TODO Pfade an AWS-Instanz anpassen
files <- data.frame(
  img = list.files(path="D:/KPopp/Documents/WWU_Muenster/Semester_7/Studienprojekt_DL_CCS/Projekt/Volcanos/etna_data/train_and_test/pixel-based/imgs/", full.names=TRUE),
  mask = list.files(path="D:/KPopp/Documents/WWU_Muenster/Semester_7/Studienprojekt_DL_CCS/Projekt/Volcanos/etna_data/train_and_test/pixel-based/masks/", full.names=TRUE)
)

# randomly split the data.frame with the file-paths into a training-dataset (75%) and a validation-dataset (25%)
# 'training(files)' will get the training-data-paths
# 'testing(files)' will get the validation-data-paths
files <- initial_split(files, prop = 0.75)
files_training = training(files)
files_validation = testing(files)
# first column: images
# second column: masks

# replace the paths with the corresponding real raster data:
# therefore make an array out of the real image and mask values (for both training and
# validation data) with function 'read_tif'
files_training$img <- lapply(files_training$img, read_tif)
files_training$img <- lapply(files_training$img, function(x){x/10000}) # rescale Sentinel-2 data to between 0 and 1
files_training$mask <- lapply(files_training$mask, read_tif, TRUE)
# TODO evtl. Reskalierung auch für Masken machen, je nachdem, ob unsere Masken bereits nur 0-1 sind
files_validation$img <- lapply(files_validation$img, read_tif)
files_validation$img <- lapply(files_validation$img, function(x){x/10000}) # rescale Sentinel-2 data to between 0 and 1
files_validation$mask <- lapply(files_validation$mask, read_tif, TRUE)
# TODO evtl. Reskalierung auch für Masken machen, je nachdem, ob unsere Masken bereits nur 0-1 sind

# prepare data for training (apply data augmentation)
training_dataset <- dl_prepare_data_tif(files_training,
                                        train = TRUE,
                                        model_input_shape = c(448,448),
                                        batch_size = 10L)
validation_dataset <- dl_prepare_data_tif(files_validation,
                                          train = FALSE,
                                          model_input_shape = c(448,448),
                                          batch_size = 10L)


### inspect the resulting data set:
# get all tensors through the python iterator
training_tensors <- training_dataset%>%as_iterator()%>%iterate()
# check that the amount of data has increased:
length(training_tensors) # number of tensors (1 tensor has 10 images as defined in ...)


######################################################################################################

# train the CNN (the network is written in "CNN_pixel-based.R") with the training data that
# was preprocessed above

# JUST USE THIS, IF THE COMPUTER WHICH HAS TO DO THE TRAINING IS ABLE TO PERFORM ON THIS
compile(
  unet_model,
  optimizer = optimizer_rmsprop(lr = 1e-5),
  loss = "binary_crossentropy",
  metrics = c(metric_binary_accuracy)
)


diagnostics <- fit(unet_model,
                   training_dataset,
                   epochs = 15,
                   validation_data = validation_dataset)
plot(diagnostics)



# compare the result to the mask on one of the validation samples:
sample <- floor(runif(n = 1,min = 1,max = 4))
img_path <- as.character(testing(files)[[sample,1]])
mask_path <- as.character(testing(files)[[sample,2]])
img <- magick::image_read(img_path)
mask <- magick::image_read(mask_path)
# 'object' is the CNN which will be used for prediction:
pred <- magick::image_read(as.raster(predict(object = unet_model, validation_dataset)[sample,,,]))

out <- magick::image_append(c(
  magick::image_append(mask, stack = TRUE),
  magick::image_append(img, stack = TRUE), 
  magick::image_append(pred, stack = TRUE)
)
)

plot(out)


######################################################################################################

# predict with the trained CNN:

### first with the test-data:

# TODO Pfad an AWS-Instanz anpassen
test_dataset <- dl_prepare_data_tif(train = FALSE,
                                    predict = TRUE,
                                    subsets_path="D:/KPopp/Documents/WWU_Muenster/Semester_7/Studienprojekt_DL_CCS/Projekt/Volcanos/etna_data/train_and_test/pixel-based/imgs/", # TODO subset_path anpassen
                                    model_input_shape = c(448,448),
                                    batch_size = 5L) # TODO evtl. batch size anpassen

system.time(predictions <- predict(unet_model, test_dataset))


# TODO
# TODO Pfad an AWS-Instanz anpassen
rebuild_img(pred_subsets = predictions,
            out_path = "D:/KPopp/Documents/WWU_Muenster/Semester_7/Studienprojekt_DL_CCS/Projekt/Volcanos/etna_data/train_and_test/pixel-based/", # dahin wird fertiges output geschrieben (ordner out wird erstellt)
            target_rst = sentinel_subsets)



### then with the real data for productive operation:

# TODO




######################################################################################################

# FOR THE PRETRAINED NET:

#compile(
#  pretrained_model,
#  optimizer = optimizer_rmsprop(lr = 1e-5),
#  loss = "binary_crossentropy",
#  metrics = c("accuracy")
#)

# auch wenn nat?rlich das ganze Netz f?rs Training durchlaufen wird, werden quasi nur 
# unsere eigenen Layer trainiert:
#diagnostics <- fit(pretrained_model,
#                   training_dataset,
#                   epochs = 6,
#                   validation_data = validation_dataset)
#plot(diagnostics)

# --> IM PLOT PRUEFEN, OB NUN GUTE ANPASSUNGEN SCHON NACH WENIGER EOPCHEN ALS MIT ERSTEM EINFACHEN NETZ
# (ob: training curve flattens at a high accuracy already after less epochs as before)

#diagnostics$metrics
