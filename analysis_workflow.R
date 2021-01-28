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
source("data_processing.R")
source("tif_processing.R")
source("CNN_pixel-based.R")


#################################### TRAINING AND VALIDATION DATA ####################################

### make subset-tiles from original Sentinel-2 image by using function 'dl_subsets' from 'handle_subsets.R':
# ETNA
etna_full <- stack(paste(getwd(), "/etna_data/etna_b2_b3_b4_b8_b12.tif", sep = ""))
etna_subsets = dl_subsets(inputrst = etna_full,
                          targetsize = c(100,100),  # TODO adapt targetsize
                          targetdir = (paste(getwd(), "/etna_data/pixel-based/train/imgs/", sep = "")),  # must already exist
                          targetname = "etna_subset_")

etna_mask <- stack(paste(getwd(), "/etna_data/etna_mask.tif", sep = ""))
etna_mask_subsets = dl_subsets(inputrst = etna_mask,
                               targetsize = c(100,100),  # TODO adapt targetsize
                               targetdir = (paste(getwd(), "/etna_data/pixel-based/train/masks/", sep = "")),
                               targetname = "etna_mask_subset_")

# SAKURAJIMA
saku_full <- stack(paste(getwd(), "/sakurajima_data/saku_b2_b3_b4_b8_b12.tif", sep = ""))
saku_subsets = dl_subsets(inputrst = saku_full,
                          targetsize = c(100,100),  # TODO adapt targetsize
                          targetdir = (paste(getwd(), "/sakurajima_data/pixel-based/train/imgs/", sep = "")),
                          targetname = "saku_subset_")

saku_mask <- stack(paste(getwd(), "/sakurajima_data/saku_mask.tif", sep = ""))
saku_mask_subsets = dl_subsets(inputrst = saku_mask,
                               targetsize = c(100,100),  # TODO adapt targetsize
                               targetdir = (paste(getwd(), "/sakurajima_data/pixel-based/train/masks/", sep = "")),
                               targetname = "saku_mask_subset_")

# TODO SUWANOSEJIMA
#suwa_full <- stack(paste(getwd(), "/suwanosejima_data/suwa_b2_b3_b4_b8_b12.tif", sep = ""))
#suwa_subsets = dl_subsets(inputrst = suwa_full,
#                          targetsize = c(100,100),  # TODO adapt targetsize
#                          targetdir = (paste(getwd(), "/suwanosejima_data/pixel-based/train/imgs/", sep = "")),
#                          targetname = "suwa_subset_")


# TODO write functionality to read 'etna_subsets' etc., needed for rebuild_img


### data preprocessing with data augmentation:

# *************************************** ETNA ***************************************
# make data.frame with full paths of etna-images and their masks
# TODO make one for both volcanos?
etna_files <- data.frame(
  img = list.files(path = (paste(getwd(), "/etna_data/pixel-based/train/imgs/", sep = "")), full.names=TRUE),
  mask = list.files(path = (paste(getwd(), "/etna_data/pixel-based/train/masks/", sep = "")), full.names=TRUE)
)

# randomly split the data.frame with the file-paths into a training-dataset (75%) and a validation-dataset (25%)
# 'training(files)' will get the training-data-paths
# 'testing(files)' will get the validation-data-paths
etna_files <- initial_split(etna_files, prop = 0.75)
etna_files_training = training(etna_files)
etna_files_validation = testing(etna_files)
# first column: images
# second column: masks
# up to now the "files" are just the paths on disk

# make training-dataset and validation-dataset of etna-data, both with data augmentation
etna_training_dataset = make_dataset_for_CNN(files = etna_files_training, train = TRUE)
etna_validation_dataset = make_dataset_for_CNN(files = etna_files_validation, train = FALSE)

inspect_both_datasets(etna_training_dataset, etna_validation_dataset)


# *************************************** SAKU ***************************************
saku_files <- data.frame(
  img = list.files(path = (paste(getwd(), "/sakurajima_data/pixel-based/train/imgs/", sep = "")), full.names=TRUE),
  mask = list.files(path = (paste(getwd(), "/sakurajima_data/pixel-based/train/masks/", sep = "")), full.names=TRUE)
)

saku_files <- initial_split(saku_files, prop = 0.75)
saku_files_training = training(saku_files)
saku_files_validation = testing(saku_files)

# make training-dataset and validation-dataset of saku-data, both with data augmentation
saku_training_dataset = make_dataset_for_CNN(files = saku_files_training, train = TRUE)
saku_validation_dataset = make_dataset_for_CNN(files = saku_files_validation, train = FALSE)

inspect_both_datasets(saku_training_dataset, saku_validation_dataset)



########################################### TRAIN THE CNN ###########################################

# the network is written in "CNN_pixel-based.R"

# JUST USE THIS WITH A GPU
compile(
  u_net,
  optimizer = optimizer_rmsprop(lr = 1e-5),  # TODO adapt learning rate
  loss = "binary_crossentropy",
  metrics = c(metric_binary_accuracy)
)

# train with etna-data:
diagnostics <- fit(u_net,
                   etna_training_dataset,
                   epochs = 17,  # TODO adapt number of epochs
                   validation_data = etna_validation_dataset)
plot(diagnostics)

# train with saku-data:
diagnostics <- fit(u_net,
                   saku_training_dataset,
                   epochs = 17,  # TODO adapt number of epochs
                   validation_data = saku_validation_dataset)
plot(diagnostics)


### inspect one result
# one result-subset of etna:
inspect_one_result_subset(files = etna_files, validation_dataset = etna_validation_dataset, max = 38)
# one result-subset of saku:
inspect_one_result_subset(files = saku_files, validation_dataset = saku_validation_dataset, max = 20)



########################################## PREDICTION ##########################################

# *************************************** ETNA ***************************************
# make subsets of the image on which to predict:
etna_full_pred <- stack(paste(getwd(), "/etna_data/suwa_b2_b3_b4_b8_b12.tif", sep = ""))  # TODO anpassen
etna_subsets_pred = dl_subsets(inputrst = suwa_full_pred,
                               targetsize = c(100,100),
                               targetdir = (paste(getwd(), "/etna_data/pixel-based/prediction/imgs/", sep = "")),  # must already exist
                               targetname = "")

# TODO
# ...

# *************************************** SAKU ***************************************
saku_full_pred <- stack(paste(getwd(), "/sakurajima_data/suwa_b2_b3_b4_b8_b12.tif", sep = ""))  # TODO anpassen
saku_subsets_pred = dl_subsets(inputrst = saku_full_pred,
                               targetsize = c(100,100),
                               targetdir = (paste(getwd(), "/sakurajima_data/pixel-based/prediction/imgs/", sep = "")),  # must already exist
                               targetname = "")

# TODO
# ...

# *************************************** SUWA ***************************************
suwa_full_pred <- stack(paste(getwd(), "/suwanosejima_data/suwa_b2_b3_b4_b8_b12.tif", sep = ""))  # TODO anpassen
suwa_subsets_pred = dl_subsets(inputrst = suwa_full_pred,
                               targetsize = c(100,100),
                               targetdir = (paste(getwd(), "/suwanosejima_data/pixel-based/prediction/imgs/", sep = "")),  # must already exist
                               targetname = "")

# TODO use function make_dataset_for_CNN OR change its signature
# predict on these subsets with the trained CNN:
suwa_prediction_dataset <- dl_prepare_data_tif(train = FALSE,
                                               predict = TRUE,
                                               subsets_path = (paste(getwd(), "/suwanosejima_data/pixel-based/prediction/imgs/", sep = "")),
                                               model_input_shape = c(100,100),  # TODO adapt model_input_shape
                                               batch_size = 10L)  # TODO adapt batch size ??

# TODO ORDERING DOES NOT WORK PROPERLY:
#order(as.numeric(tools::file_path_sans_ext(basename(list.files((paste(getwd(), "/suwanosejima_data/pixel-based/prediction/imgs/", sep = "")))))))

system.time(suwa_predictions <- predict(u_net, suwa_prediction_dataset))
save_model_hdf5(u_net, filepath = "./u_net.h5")

# assemble the predictions:
rebuild_img(pred_subsets = suwa_predictions,
            out_path = (paste(getwd(), "/suwanosejima_data/pixel-based/prediction/", sep = "")),  # here the output will be written (folder 'out' will be created)
            target_rst = suwa_subsets_pred)  # output of 'dl_subsets'



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
