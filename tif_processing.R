# Code for study project "Einsatz von Cloud Computing Services fuer Deep Learning in der Fernerkundung"
# in the winter semester 2020/2021 at the University of Muenster
# by Fabian Fermazin and Katharina Poppinga

reduce_channels <- function(arr, channels) {
  arr = arr[,,(channels)]
  return (arr)
}


# the following two functions are gotten from https://pad.uni-muenster.de/***
# author: Christian Knoth
# uses code from: https://blogs.rstudio.com/ai/posts/2019-08-23-unet/ (accessed 2020-08-12)

# loads a TIF-file and makes an array out of it
# with package 'stars'
read_tif <- function(f, mask=FALSE) {
  out = array(NA)
  out = unclass(read_stars(f))[[1]]
  if(mask==T) { # add a dummy dimension
    dim(out) <- c(dim(out), 1)
  }
  return(out)
}


# preprocessing of TIF-files given in data.frames (arrays)
dl_prepare_data_tif <- function(files, train, predict=FALSE, subsets_path=NULL, model_input_shape = c(448,448), batch_size = 10L) {
  
  ###### preparing training or validation data: ######
  if (!predict) {
    # function for random change of saturation, brightness and hue, will be used as part of the augmentation
    spectral_augmentation <- function(img) {
      img %>%
        tf$image$random_brightness(max_delta = 0.3) %>%
        tf$image$random_contrast(lower = 0.5, upper = 0.7) %>%
        # TODO the following is not supported for >3 bands - you can uncomment in case you use only 3band images
        #tf$image$random_saturation(lower = 0.5, upper = 0.7) %>%
        # make sure we still are between 0 and 1
        tf$clip_by_value(0, 1)
    }
    
    # array to tensor: create a tf_dataset from the first two coloumns of data.frame (ignoring area number used for splitting during data preparation),
    # TODO das stimmt nicht?: überprüfen: "right now still containing only paths to images"
    dataset <- tensor_slices_dataset(files[,1:2])
    
    #convert to float32:
    #for each record in dataset, both its list items are modyfied by the result of applying convert_image_dtype to them
    dataset <- dataset_map(dataset, function(.x) list_modify(.x,
                                                             img = tf$image$convert_image_dtype(.x$img, dtype = tf$float64),
                                                             mask = tf$image$convert_image_dtype(.x$mask, dtype = tf$float64)
    ))
    
    # resize:
    # for each record in dataset, both its list items are modified by the results of applying resize to them
    dataset <-
      dataset_map(dataset, function(.x)
        list_modify(.x, img = tf$image$resize(.x$img, size = shape(model_input_shape[1], model_input_shape[2])),
                    mask = tf$image$resize(.x$mask, size = shape(model_input_shape[1], model_input_shape[2]))))
    
    # data augmentation performed on training set only
    if (train) {
      # augmentation 1: flip left right, including random change of saturation, brightness and contrast
      # for each record in dataset, only the img item is modified by the result of applying spectral_augmentation to it
      augmentation <- dataset_map(dataset, function(.x) list_modify(.x,
                                                                    img = spectral_augmentation(.x$img)
      ))
      # ...as opposed to this, flipping is applied to img and mask of each record
      augmentation <- dataset_map(augmentation, function(.x) list_modify(.x,
                                                                         img = tf$image$flip_left_right(.x$img),
                                                                         mask = tf$image$flip_left_right(.x$mask)
      ))
      dataset_augmented <- dataset_concatenate(dataset, augmentation)
      
      # augmentation 2: flip up down, including random change of saturation, brightness and contrast
      augmentation <- dataset_map(dataset, function(.x) list_modify(.x,
                                                                    img = spectral_augmentation(.x$img)
      ))
      augmentation <- dataset_map(augmentation, function(.x) list_modify(.x,
                                                                         img = tf$image$flip_up_down(.x$img),
                                                                         mask = tf$image$flip_up_down(.x$mask)
      ))
      dataset_augmented <- dataset_concatenate(dataset_augmented, augmentation)
      
      # augmentation 3: flip left right AND up down, including random change of saturation, brightness and contrast
      augmentation <- dataset_map(dataset, function(.x) list_modify(.x,
                                                                    img = spectral_augmentation(.x$img)
      ))
      augmentation <- dataset_map(augmentation, function(.x) list_modify(.x,
                                                                         img = tf$image$flip_left_right(.x$img),
                                                                         mask = tf$image$flip_left_right(.x$mask)
      ))
      augmentation <- dataset_map(augmentation, function(.x) list_modify(.x,
                                                                         img = tf$image$flip_up_down(.x$img),
                                                                         mask = tf$image$flip_up_down(.x$mask)
      ))
      dataset_augmented <- dataset_concatenate(dataset_augmented, augmentation)
    }
    
    # shuffling on training set only
    if (train) {
      dataset <- dataset_shuffle(dataset_augmented, buffer_size = batch_size*128)
    }
    
    # train in batches; batch size might need to be adapted depending on
    # available memory
    dataset <- dataset_batch(dataset, batch_size)
    
    # output needs to be unnamed
    dataset <- dataset_map(dataset, unname)
  }
  
  ###### preparing data for real prediction (no validation and no data augmentation): ######
  else {
    # make sure subsets are read in correct order so that they can later be reassambled correctly
    # needs files to be named accordingly (only number)
    o <- order(as.numeric(tools::file_path_sans_ext(basename(list.files(subsets_path)))))
    subset_list <- list.files(subsets_path, full.names = T)[o]
    
    dataset <- tensor_slices_dataset(subset_list)
    dataset <- dataset_map(dataset, function(.x) tf$image$convert_image_dtype(.x, dtype = tf$float32))
    dataset <- dataset_map(dataset, function(.x) tf$image$resize(.x, size = shape(model_input_shape[1], model_input_shape[2])))
    dataset <- dataset_batch(dataset, batch_size)
    dataset <- dataset_map(dataset, unname)
  }
}
