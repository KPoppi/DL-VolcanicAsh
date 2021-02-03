# Code for study project "Einsatz von Cloud Computing Services fuer Deep Learning in der Fernerkundung"
# in the winter semester 2020/2021 at the University of Muenster
# by Fabian Fermazin and Katharina Poppinga

# this file contains the convolutional neural network which is used for the
# deep learning analysis of vulcano ash deposition

# u_net with 3 convolutional-layer per block 


############################### U-NET FOR PIXEL-BASED ANALYSIS (u_net) ###############################

# uses functional API from Keras:

### "contracting path" ###

# input
input_tensor <- layer_input(shape = shape)

#conv block 1
unet_tensor <- layer_conv_2d(input_tensor, filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu")
conc_tensor2 <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu")
unet_tensor <- layer_max_pooling_2d(conc_tensor2)

#conv block 2
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3), padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3), padding = "same", activation = "relu")
conc_tensor1 <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3), padding = "same", activation = "relu")
unet_tensor <- layer_max_pooling_2d(conc_tensor1)

#"bottom curve" of unet
unet_tensor <- layer_conv_2d(unet_tensor, filters = 256, kernel_size = c(3,3), padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 256, kernel_size = c(3,3), padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 256, kernel_size = c(3,3), padding = "same", activation = "relu")


### expanding path ###

# upsampling block 1
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 128, kernel_size = c(2,2), strides = 2, padding = "same")
unet_tensor <- layer_concatenate(list(conc_tensor1, unet_tensor)) # for resampling back to finer grained resolution
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3), padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3), padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3), padding = "same", activation = "relu")

# upsampling block 2
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 64, kernel_size = c(2,2), strides = 2, padding = "same")
unet_tensor <- layer_concatenate(list(conc_tensor2, unet_tensor)) # for resampling back to finer grained resolution
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu")

# output
unet_tensor <- layer_conv_2d(unet_tensor, filters = 1, kernel_size = 1, activation = "sigmoid")


### combine final unet_tensor (carrying all the transformations applied through the layers)
### with the input_tensor to create the whole U-net model
u_net <- keras_model(inputs = input_tensor, outputs = unet_tensor)

u_net
