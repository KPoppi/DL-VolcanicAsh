# Code for study project "Einsatz von Cloud Computing Services fuer Deep Learning in der Fernerkundung"
# in the winter semester 2020/2021 at the University of Muenster
# by Fabian Fermazin and Katharina Poppinga

#install.packages(c("reticulate", "tensorflow", "keras", "purrr", "rsample", "abind", "sf", "stars", "tfdatasets", "digest", "ggplot2", "sp", "raster", "mapview", "gdalUtils", "magick"))

libraries = c("reticulate", "tensorflow", "keras", "purrr", "rsample", "abind", "sf", "stars", "tfdatasets", "digest", "ggplot2", "sp", "raster", "mapview", "gdalUtils", "magick")
lapply(libraries, require, character.only = TRUE)

#reticulate::install_miniconda()
#keras::install_keras()

reticulate::py_config() # "reticulate::" replaces"library(reticulate)"
tensorflow::tf_config()
keras::is_keras_available()

setwd("/home/sp_gruppe2/StudyProject/DL-VolcanicAsh") # path for AWS-instance
getwd()

source("handle_subsets.R")
source("tif_processing.R")
source("CNN_pixel-based.R")

# TODO make paths more flexible for all volcanos


# *****************************************************************************************************

#################################### TRAINING AND VALIDATION DATA ####################################

# TODO write the following into function that could be called with a specific volcano

### make subset-tiles from original Sentinel-2 image by using function 'dl_subsets' from 'handle_subsets.R':

etna_full <- stack(paste(getwd(), "/etna_data/etna_b2_b3_b4_b8_b12.tif", sep = ""))
etna_subsets = dl_subsets(inputrst = etna_full,
                          targetsize = c(100,100),  # TODO adapt targetsize
                          targetdir = (paste(getwd(), "/etna_data/pixel-based/train/imgs/", sep = "")),  # must already exist
                          targetname = "etna_subset_")

etna_mask <- stack(paste(getwd(), "/etna_data/etna_mask.tif", sep = ""))
etna_mask_subsets = dl_subsets(inputrst = etna_mask,
                               targetsize = c(100,100),  # TODO adapt targetsize
                               targetdir = (paste(getwd(), "/etna_data/pixel-based/train/masks/", sep = "")),  # must already exist
                               targetname = "etna_mask_subset_")

# TODO saku_full <- stack(paste(getwd(), "/etna_data/saku_b2_b3_b4_b8_b12.tif", sep = ""))
# TODO suwa_full <- stack(paste(getwd(), "/etna_data/suwa_b2_b3_b4_b8_b12.tif", sep = ""))


# TODO write functionality to read 'etna_subsets', needed for rebuild_img


### data preprocessing with data augmentation:

# make data.frame with full paths of images and their masks
files <- data.frame(
  img = list.files(path = (paste(getwd(), "/etna_data/pixel-based/train/imgs/", sep = "")), full.names=TRUE),
  mask = list.files(path = (paste(getwd(), "/etna_data/pixel-based/train/masks/", sep = "")), full.names=TRUE)
)

# randomly split the data.frame with the file-paths into a training-dataset (75%) and a validation-dataset (25%)
# 'training(files)' will get the training-data-paths
# 'testing(files)' will get the validation-data-paths
files <- initial_split(files, prop = 0.75)
files_training = training(files)
files_validation = testing(files)
# first column: images
# second column: masks
# up to now the "files" are just the paths on disk

# replace the paths with the corresponding real raster data:
# therefore make an array out of the real image and mask values (for both training and
# validation data) with function 'read_tif'
files_training$img <- lapply(files_training$img, read_tif)
files_training$img <- lapply(files_training$img, function(x){x/10000})  # rescale Sentinel-2 data to between 0 and 1
files_training$mask <- lapply(files_training$mask, read_tif, TRUE)
files_validation$img <- lapply(files_validation$img, read_tif)
files_validation$img <- lapply(files_validation$img, function(x){x/10000})  # rescale Sentinel-2 data to between 0 and 1
files_validation$mask <- lapply(files_validation$mask, read_tif, TRUE)


# prepare data for training (apply data augmentation)
training_dataset <- dl_prepare_data_tif(files_training,
                                        train = TRUE,
                                        model_input_shape = c(100,100),
                                        batch_size = 10L)
validation_dataset <- dl_prepare_data_tif(files_validation,
                                          train = FALSE,
                                          model_input_shape = c(100,100),
                                          batch_size = 10L)


### inspect the resulting data set:

# get all tensors through the python iterator
training_tensors <- training_dataset%>%as_iterator()%>%iterate()
validation_tensors <- validation_dataset%>%as_iterator()%>%iterate()
# check that the amount of data has increased:
length(training_tensors) # number of tensors (1 tensor has 10 images as defined by 'batch_size' above)
length(validation_tensors)

dataset_iterator <- as_iterator(training_dataset)
dataset_list <- iterate(dataset_iterator)
dataset_list[[1]][[1]]

dataset_iterator <- as_iterator(validation_dataset)
dataset_list <- iterate(dataset_iterator)
dataset_list[[1]][[1]]

# *****************************************************************************************************

########################################### TRAIN THE CNN ###########################################

# the network is written in "CNN_pixel-based.R"

# JUST USE THIS WITH A GPU
compile(
  u_net,
  optimizer = optimizer_rmsprop(lr = 1e-5),
  loss = "binary_crossentropy",
  metrics = c(metric_binary_accuracy)
)

diagnostics <- fit(u_net,
                   training_dataset,
                   epochs = 20,  # TODO adapt number of epochs
                   validation_data = validation_dataset)
plot(diagnostics)



# compare the result to the mask on one of the validation samples:
sample <- floor(runif(n = 1,min = 1,max = 38))
img_path <- as.character(testing(files)[[sample,1]])
mask_path <- as.character(testing(files)[[sample,2]])
pimg <- read_tif(img_path)
img <- magick::image_read(normalize_tiff(pimg[,,c(3,2,1)]))
mask <- magick::image_read(mask_path)
# 'object' is the CNN which will be used for prediction:
pred <- magick::image_read(as.raster(predict(object = u_net, validation_dataset)[sample,,,]))

out <- magick::image_append(c(
  magick::image_append(mask, stack = TRUE),
  magick::image_append(img, stack = TRUE),
  magick::image_append(pred, stack = TRUE)
)
)

plot(out)


# *****************************************************************************************************

########################################## PREDICTION DATA ##########################################

# make subsets of the image on which to predict:
etna_full <- stack(paste(getwd(), "/etna_data/etna_b2_b3_b4_b8_b12.tif", sep = ""))
etna_subsets = dl_subsets(inputrst = etna_full,
                          targetsize = c(448,448),
                          targetdir = (paste(getwd(), "/etna_data/pixel-based/prediction/imgs/", sep = "")),  # must already exist
                          targetname = "etna_subset_")


#################################### PREDICT WITH THE TRAINED CNN ####################################

# predict on these subsets with the trained CNN:
prediction_dataset <- dl_prepare_data_tif(train = FALSE,
                                          predict = TRUE,
                                          subsets_path = (paste(getwd(), "/etna_data/pixel-based/prediction/imgs/", sep = "")),
                                          model_input_shape = c(448,448),  # TODO adapt model_input_shape
                                          batch_size = 5L)  # TODO adapt batch size ??

system.time(predictions <- predict(u_net, prediction_dataset))
save_model_hdf5(u_net, filepath = "./u_net.h5")

# assemble the predictions:
rebuild_img(pred_subsets = predictions,
            out_path = (paste(getwd(), "/etna_data/pixel-based/prediction/", sep = "")),  # here the output will be written (folder 'out' will be created)
            target_rst = etna_subsets)  # output of 'dl_subsets'


# *****************************************************************************************************


######################################################################################################

# using the 'pretrained_net':
# TODO number of bands must be adapted because pretrained_net takes just 3 bands

compile(
  pretrained_net,
  optimizer = optimizer_rmsprop(lr = 1e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# the whole net will be used while training but only the weights of our own added layers will be adapted
diagnostics <- fit(pretrained_net,
                   training_dataset,
                   epochs = 6,
                   validation_data = validation_dataset)
plot(diagnostics)

# --> IM PLOT PRUEFEN, OB NUN GUTE ANPASSUNGEN SCHON NACH WENIGER EPOCHEN ALS MIT ERSTEM EINFACHEN NETZ
# (ob: training curve flattens at a high accuracy already after less epochs as before)

diagnostics$metrics
