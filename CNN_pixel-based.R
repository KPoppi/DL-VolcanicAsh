# Code for study project "Einsatz von Cloud Computing Services fuer Deep Learning in der Fernerkundung"
# in the winter semester 2020/2021 at the University of Muenster
# by Fabian Fermazin and Katharina Poppinga

# this file contains the convolutional neural network which is used for the
# deep learning analysis of vulcano ash deposition

# 3 network-models:
# u_net
# pretrained_net (only tile-based, not-pixel based)
# combined_u_net


############################### U-NET FOR PIXEL-BASED ANALYSIS (u_net) ###############################

# uses functional API from Keras:

### "contracting path" ###

# input
input_tensor <- layer_input(shape = c(448,448,5))

#conv block 1
unet_tensor <- layer_conv_2d(input_tensor, filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu")
conc_tensor2 <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu")
unet_tensor <- layer_max_pooling_2d(conc_tensor2)

#conv block 2
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3), padding = "same", activation = "relu")
conc_tensor1 <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3), padding = "same", activation = "relu")
unet_tensor <- layer_max_pooling_2d(conc_tensor1)

#"bottom curve" of unet
unet_tensor <- layer_conv_2d(unet_tensor,filters = 256, kernel_size = c(3,3), padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor,filters = 256, kernel_size = c(3,3), padding = "same", activation = "relu")


### expanding path ###

# upsampling block 1
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 128, kernel_size = c(2,2), strides = 2, padding = "same") 
unet_tensor <- layer_concatenate(list(conc_tensor1, unet_tensor)) # for resampling back to finer grained resolution 
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3), padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3), padding = "same", activation = "relu")

# upsampling block 2
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 64, kernel_size = c(2,2), strides = 2, padding = "same")
unet_tensor <- layer_concatenate(list(conc_tensor2, unet_tensor)) # for resampling back to finer grained resolution
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = c(3,3), padding = "same", activation = "relu")

# output
unet_tensor <- layer_conv_2d(unet_tensor, filters = 1, kernel_size = 1, activation = "sigmoid")


### combine final unet_tensor (carrying all the transformations applied through the layers) 
### with the input_tensor to create the whole U-net model
u_net <- keras_model(inputs = input_tensor, outputs = unet_tensor)

u_net



################################ PRETRAINED CNN: VGG16 (pretrained_net) ################################

# NOT PIXEL-BASED

# load vgg16 as basis for feature extraction
vgg16_feat_extr <- application_vgg16(include_top = F,
                                     input_shape = c(448,448,3),
                                     weights = "imagenet")
# freeze weights, for not updating the weights (that are already adapted) again
freeze_weights(vgg16_feat_extr)
# only use layers 1 to 15
pretrained_net <- keras_model_sequential(vgg16_feat_extr$layers[1:15])

# add own flatten and dense layers for the classification 
# these dense layers are going to be trained on our own data only
pretrained_net <- layer_flatten(pretrained_net)
pretrained_net <- layer_dense(pretrained_net, units = 256, activation = "relu")
pretrained_net <- layer_dense(pretrained_net, units = 1, activation = "sigmoid")

pretrained_net



############### PRETRAINED VGG16 COMBINED WITH A DIFFERENT OWN U-NET (combined_u_net) ###############

# load pretrained vgg16 and use part of it as contracting path (feature extraction)
vgg16_feat_extr <- application_vgg16(weights = "imagenet", include_top = FALSE, input_shape = c(448,448,3))

# optionally freeze first layers to prevent changing of their weights, either whole convbase or only certain layers
# freeze_weights(vgg16_feat_extr) #or:
# freeze_weights(vgg16_feat_extr, to = "block1_pool") 

# do not use the whole model but only up to layer 15
unet_tensor <- vgg16_feat_extr$layers[[15]]$output


### add the second part of 'U' for segmentation ###

# "bottom curve" of U-net
unet_tensor <- layer_conv_2d(unet_tensor, filters = 1024, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 1024, kernel_size = 3, padding = "same", activation = "relu")

# upsampling block 1
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 512, kernel_size = 2, strides = 2, padding = "same")
unet_tensor <- layer_concatenate(list(vgg16_feat_extr$layers[[14]]$output, unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 512, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 512, kernel_size = 3, padding = "same", activation = "relu")

# upsampling block 2
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 256, kernel_size = 2, strides = 2, padding = "same")
unet_tensor <- layer_concatenate(list(vgg16_feat_extr$layers[[10]]$output, unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor,filters = 256, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor,filters = 256, kernel_size = 3, padding = "same", activation = "relu")

# upsampling block 3
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 128, kernel_size = 2, strides = 2, padding = "same")
unet_tensor <- layer_concatenate(list(vgg16_feat_extr$layers[[6]]$output, unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = 3, padding = "same", activation = "relu")

# upsampling block 4
unet_tensor <- layer_conv_2d_transpose(unet_tensor, filters = 64, kernel_size = 2, strides = 2, padding = "same")
unet_tensor <- layer_concatenate(list(vgg16_feat_extr$layers[[3]]$output, unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = 3, padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = 3, padding = "same", activation = "relu")

# final output
unet_tensor <- layer_conv_2d(unet_tensor, filters = 1, kernel_size = 1, activation = "sigmoid")

# create model from tensors
combined_u_net <- keras_model(inputs = vgg16_feat_extr$input,
                                   outputs = unet_tensor)

combined_u_net
