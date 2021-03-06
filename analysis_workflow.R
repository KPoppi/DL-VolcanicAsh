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

load("~/StudyProject/DL-VolcanicAsh/workspace.RData")
rm(u_net)

#size = c(120,120)
size = c(100,100)
#shape = c(120,120,5)
shape = c(100,100,5)

source("handle_subsets.R")
source("data_processing.R")
source("tif_processing.R")
#source("CNN.R")
source("CNN_2.R")

#u_net <- load_model_hdf5("./u_net_etna.h5")
#u_net <- load_model_hdf5("./u_net_saku.h5")
#u_net <- load_model_hdf5("./u_net_etna_saku.h5")

#################################### TRAINING AND VALIDATION DATA ####################################

### make subset-tiles from original Sentinel-2 image by using function 'dl_subsets' from 'handle_subsets.R':
# ETNA
etna_full <- stack(paste(getwd(), "/etna_data/etna_b2_b3_b4_b8_b12.tif", sep = ""))
etna_subsets = dl_subsets(inputrst = etna_full,
                          targetsize = size,
                          targetdir = (paste(getwd(), "/etna_data/pixel-based/train/imgs/", sep = "")),  # must already exist
                          targetname = "etna_subset_")

etna_mask <- stack(paste(getwd(), "/etna_data/etna_mask.tif", sep = ""))
etna_mask_subsets = dl_subsets(inputrst = etna_mask,
                               targetsize = size,
                               targetdir = (paste(getwd(), "/etna_data/pixel-based/train/masks/", sep = "")),
                               targetname = "etna_mask_subset_")

# 70% ETNA
etna_full <- stack(paste(getwd(), "/etna_data/etna_b2_b3_b4_b8_b12_train_part.tif", sep = ""))
etna_subsets = dl_subsets(inputrst = etna_full,
                          targetsize = size,
                          targetdir = (paste(getwd(), "/etna_data/pixel-based/train_part/imgs/", sep = "")),  # must already exist
                          targetname = "etna_subset_")

etna_mask <- stack(paste(getwd(), "/etna_data/etna_mask_train_part.tif", sep = ""))
etna_mask_subsets = dl_subsets(inputrst = etna_mask,
                               targetsize = size,
                               targetdir = (paste(getwd(), "/etna_data/pixel-based/train_part/masks/", sep = "")),
                               targetname = "etna_mask_subset_")

# SAKURAJIMA
saku_full <- stack(paste(getwd(), "/sakurajima_data/saku_b2_b3_b4_b8_b12.tif", sep = ""))
saku_subsets = dl_subsets(inputrst = saku_full,
                          targetsize = size,
                          targetdir = (paste(getwd(), "/sakurajima_data/pixel-based/train/imgs/", sep = "")),
                          targetname = "saku_subset_")

saku_mask <- stack(paste(getwd(), "/sakurajima_data/saku_mask.tif", sep = ""))
saku_mask_subsets = dl_subsets(inputrst = saku_mask,
                               targetsize = size,
                               targetdir = (paste(getwd(), "/sakurajima_data/pixel-based/train/masks/", sep = "")),
                               targetname = "saku_mask_subset_")

# KILAUEA
hawa_full <- stack(paste(getwd(), "/hawaii_data/hawaii_b2_b3_b4_b8_b12.tif", sep = ""))
hawa_subsets = dl_subsets(inputrst = hawa_full,
                          targetsize = size,
                          targetdir = (paste(getwd(), "/hawaii_data/pixel-based/train/imgs/", sep = "")),
                          targetname = "hawaii_subset_")

hawa_mask <- stack(paste(getwd(), "/hawaii_data/hawaii_mask.tif", sep = ""))
hawa_mask_subsets = dl_subsets(inputrst = hawa_mask,
                               targetsize = size,
                               targetdir = (paste(getwd(), "/hawaii_data/pixel-based/train/masks/", sep = "")),
                               targetname = "hawaii_mask_subset_")


### data preprocessing with data augmentation:

# *************************************** ETNA ***************************************
# make data.frame with full paths of etna-images and their masks
etna_files <- data.frame(
  img = list.files(path = (paste(getwd(), "/etna_data/pixel-based/train/imgs/", sep = "")), full.names=TRUE),
  mask = list.files(path = (paste(getwd(), "/etna_data/pixel-based/train/masks/", sep = "")), full.names=TRUE)
)
# with 70% of etna-data:
etna_files <- data.frame(
  img = list.files(path = (paste(getwd(), "/etna_data/pixel-based/train_part/imgs/", sep = "")), full.names=TRUE),
  mask = list.files(path = (paste(getwd(), "/etna_data/pixel-based/train_part/masks/", sep = "")), full.names=TRUE)
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

# *************************************** KILAUEA ***************************************
hawa_files <- data.frame(
  img = list.files(path = (paste(getwd(), "/hawaii_data/pixel-based/train/imgs/", sep = "")), full.names=TRUE),
  mask = list.files(path = (paste(getwd(), "/hawaii_data/pixel-based/train/masks/", sep = "")), full.names=TRUE)
)

hawa_files <- initial_split(hawa_files, prop = 0.75)
hawa_files_training = training(hawa_files)
hawa_files_validation = testing(hawa_files)

# make training-dataset and validation-dataset of kilauea-data, both with data augmentation
hawa_training_dataset = make_dataset_for_CNN(files = hawa_files_training, train = TRUE)
hawa_validation_dataset = make_dataset_for_CNN(files = hawa_files_validation, train = FALSE)

inspect_both_datasets(hawa_training_dataset, hawa_validation_dataset)


########################################### TRAIN THE CNN ###########################################

# the network is written in "CNN_pixel-based.R"

# JUST USE THIS WITH A GPU
compile(
  u_net,
  optimizer = optimizer_rmsprop(lr = 1e-5),
  loss = "binary_crossentropy",
  metrics = c(metric_binary_accuracy)
)

# train with etna-data:
diagnostics <- fit(u_net,
                   etna_training_dataset,
                   epochs = 17,
                   validation_data = etna_validation_dataset)
plot(diagnostics)

save_model_hdf5(u_net, filepath = "./u_net_etna.h5")

# train with saku-data:
diagnostics <- fit(u_net,
                   saku_training_dataset,
                   epochs = 17,
                   validation_data = saku_validation_dataset)
plot(diagnostics)

# train with kilauea-data:
diagnostics <- fit(u_net,
                   hawa_training_dataset,
                   epochs = 17,
                   validation_data = hawa_validation_dataset)
plot(diagnostics)

save_model_hdf5(u_net, filepath = "./u_net_saku.h5")
save_model_hdf5(u_net, filepath = "./u_net_etna_saku.h5")
save_model_hdf5(u_net, filepath = "./u_net_etna_hawa.h5")
save_model_hdf5(u_net, filepath = "./u_net_hawa.h5")

### inspect one result
# one result-subset of etna:
inspect_one_result_subset(files = etna_files, validation_dataset = etna_validation_dataset, max = 38)
# one result-subset of saku:
inspect_one_result_subset(files = saku_files, validation_dataset = saku_validation_dataset, max = 20)



########################################## PREDICTION ##########################################

# *************************************** ETNA ***************************************
# make subsets of the image on which to predict:
# (just once but this variable 'etna_subsets_pred' is needed for reassembling the predictions)
etna_full_pred <- stack(paste(getwd(), "/etna_data/etna_b2_b3_b4_b8_b12.tif", sep = ""))
etna_subsets_pred = dl_subsets(inputrst = etna_full_pred,
                               targetsize = size,
                               targetdir = (paste(getwd(), "/etna_data/pixel-based/prediction/imgs/", sep = "")),  # must already exist
                               targetname = "")
# with 30% of etna-data:
etna_full_pred <- stack(paste(getwd(), "/etna_data/etna_b2_b3_b4_b8_b12_predict_part.tif", sep = ""))
etna_subsets_pred = dl_subsets(inputrst = etna_full_pred,
                               targetsize = size,
                               targetdir = (paste(getwd(), "/etna_data/pixel-based/prediction/imgs/", sep = "")),  # must already exist
                               targetname = "")

# read and order subsets (their paths) in ascending order (order is needed for 'rebuild_img'-function) and write them into dataframe:
etna_subsets_path = (paste(getwd(), "/etna_data/pixel-based/prediction/imgs/", sep = ""))
o <- order(as.numeric(tools::file_path_sans_ext(basename(list.files(etna_subsets_path)))))
etna_files_pred <- data.frame(
  img = list.files(etna_subsets_path, full.names = T)[o]
)

# make prediction-dataset out of those subset-paths, without data augmentation
etna_prediction_dataset = make_dataset_for_CNN(files = etna_files_pred, train = FALSE, predict = TRUE)

# predict on this dataset with the trained CNN:
system.time(etna_predictions <- predict(u_net,
                                        etna_prediction_dataset))

# reassemble the predictions:
rebuild_img(pred_subsets = etna_predictions,
            out_path = (paste(getwd(), "/etna_data/pixel-based/prediction/", sep = "")),  # here the output will be written (folder 'out' will be created)
            target_rst = etna_subsets_pred)  # output of 'dl_subsets'


# *************************************** SAKU ***************************************
saku_full_pred <- stack(paste(getwd(), "/sakurajima_data/saku_b2_b3_b4_b8_b12.tif", sep = ""))
saku_subsets_pred = dl_subsets(inputrst = saku_full_pred,
                               targetsize = size,
                               targetdir = (paste(getwd(), "/sakurajima_data/pixel-based/prediction/imgs/", sep = "")),  # must already exist
                               targetname = "")

saku_subsets_path = (paste(getwd(), "/sakurajima_data/pixel-based/prediction/imgs/", sep = ""))
o <- order(as.numeric(tools::file_path_sans_ext(basename(list.files(saku_subsets_path)))))
saku_files_pred <- data.frame(
  img = list.files(saku_subsets_path, full.names = T)[o]
)

saku_prediction_dataset = make_dataset_for_CNN(files = saku_files_pred, train = FALSE, predict = TRUE)

system.time(saku_predictions <- predict(u_net,
                                        saku_prediction_dataset))

# reassemble the predictions:
rebuild_img(pred_subsets = saku_predictions,
            out_path = (paste(getwd(), "/sakurajima_data/pixel-based/prediction/", sep = "")),
            target_rst = saku_subsets_pred)


# *************************************** SUWA ***************************************
suwa_full_pred <- stack(paste(getwd(), "/suwanosejima_data/suwa_b2_b3_b4_b8_b12.tif", sep = ""))
suwa_subsets_pred = dl_subsets(inputrst = suwa_full_pred,
                               targetsize = size,
                               targetdir = (paste(getwd(), "/suwanosejima_data/pixel-based/prediction/imgs/", sep = "")),
                               targetname = "")

suwa_subsets_path = (paste(getwd(), "/suwanosejima_data/pixel-based/prediction/imgs/", sep = ""))
o <- order(as.numeric(tools::file_path_sans_ext(basename(list.files(suwa_subsets_path)))))
suwa_files_pred <- data.frame(
  img = list.files(suwa_subsets_path, full.names = T)[o]
)

suwa_prediction_dataset = make_dataset_for_CNN(files = suwa_files_pred, train = FALSE, predict = TRUE)

system.time(suwa_predictions <- predict(u_net,
                                        suwa_prediction_dataset))

# reassemble the predictions:
rebuild_img(pred_subsets = suwa_predictions,
            out_path = (paste(getwd(), "/suwanosejima_data/pixel-based/prediction/", sep = "")),
            target_rst = suwa_subsets_pred)
