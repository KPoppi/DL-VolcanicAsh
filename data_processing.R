# Code for study project "Einsatz von Cloud Computing Services fuer Deep Learning in der Fernerkundung"
# in the winter semester 2020/2021 at the University of Muenster
# by Fabian Fermazin and Katharina Poppinga


# takes file-paths of images and prepares the dataset with the corresponding TIFs
# input-parameter: train and predict are boolean
# applies data augmentation if train = TRUE
make_dataset_for_CNN <- function(files, train, predict = FALSE, subsets_path = NULL) {
  
  # replace the paths with the corresponding real raster data:
  # therefore make an array out of the real image and mask values with function 'read_tif'
  files$img <- lapply(files$img, read_tif)
  files$img <- lapply(files$img, function(x){x/10000})  # rescale Sentinel-2 data to between 0 and 1
  if (predict == FALSE) {
    files$mask <- lapply(files$mask, read_tif, TRUE) 
  }
  
  # prepare data for training or validation (apply data augmentation)
  dataset <- dl_prepare_data_tif(files,
                                 train = train,
                                 predict = predict,
                                 model_input_shape = c(100,100),
                                 batch_size = 10L)
  return(dataset)
}


### inspect the resulting data set:
inspect_both_datasets <- function(training_dataset, validation_dataset) {

  dataset_iterator <- as_iterator(training_dataset)
  dataset_list <- iterate(dataset_iterator)
  print(dataset_list[[1]][[1]])
  dataset_iterator <- as_iterator(validation_dataset)
  dataset_list <- iterate(dataset_iterator)
  print(dataset_list[[1]][[1]])

  # get all tensors through the python iterator
  training_tensors <- training_dataset%>%as_iterator()%>%iterate()
  validation_tensors <- validation_dataset%>%as_iterator()%>%iterate()
  print(training_tensors)
  print(validation_tensors)
  
  # check that the amount of data has increased:
  print(length(training_tensors)) # number of tensors (1 tensor has 10 images as defined by 'batch_size' above)
  print(length(validation_tensors))
}


# plots one random mask-subset with its corresponding image subset
# and their corresponding prediction-result-subset
inspect_one_result_subset <- function(files, validation_dataset, max) {
  # compare the result to the mask on one of the validation samples:
  sample <- floor(runif(n = 1, min = 1, max = max))
  img_path <- as.character(testing(files)[[sample,1]])
  mask_path <- as.character(testing(files)[[sample,2]])
  pimg <- read_tif(img_path)
  img <- magick::image_read(normalize_tif(pimg[,,c(3,2,1)]))
  mask <- magick::image_read(mask_path)
  # 'object' is the CNN which will be used for prediction:
  pred <- magick::image_read(as.raster(predict(object = u_net, validation_dataset)[sample,,,]))
  
  out <- magick::image_append(c(
    magick::image_append(mask, stack = TRUE),
    magick::image_append(img, stack = TRUE),
    magick::image_append(pred, stack = TRUE)
  ))
  plot(out)
}
