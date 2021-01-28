# Code for study project "Einsatz von Cloud Computing Services fuer Deep Learning in der Fernerkundung"
# in the winter semester 2020/2021 at the University of Muenster
# by Fabian Fermazin and Katharina Poppinga


# TODO desc
# input-parameter: train and predict are boolean
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