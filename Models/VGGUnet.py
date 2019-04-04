from keras.models import *
from keras.layers import *


import os
file_path = os.path.dirname( os.path.abspath(__file__) )


VGG_Weights_path = file_path+"/../data/vgg16_weights_th_dim_ordering_th_kernels.h5"

IMAGE_ORDERING = 'channels_last'

def get_crop_shape(target, refer):
	# width, the 3rd dimension
	cw = (target.get_shape()[2] - refer.get_shape()[2]).value
	assert (cw >= 0)
	if cw % 2 != 0:
		cw1, cw2 = int(cw / 2), int(cw / 2) + 1
	else:
		cw1, cw2 = int(cw / 2), int(cw / 2)
	# height, the 2nd dimension
	ch = (target.get_shape()[1] - refer.get_shape()[1]).value
	assert (ch >= 0)
	if ch % 2 != 0:
		ch1, ch2 = int(ch / 2), int(ch / 2) + 1
	else:
		ch1, ch2 = int(ch / 2), int(ch / 2)

	return (ch1, ch2), (cw1, cw2)

def VGGUnet( n_classes ,  input_height=256, input_width=256 , vgg_level=3):
	concat_axis = 3
	assert input_height%32 == 0
	assert input_width%32 == 0

	# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
	img_input = Input(shape=(input_height,input_width,3))

	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
	f1 = x
	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
	f5 = x

	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dense( 1000 , activation='softmax', name='predictions')(x)

	vgg  = Model(  img_input , x  )
	#vgg.load_weights(VGG_Weights_path)

	levels = [f1 , f2 , f3 , f4 , f5 ]

	o = f4

	o = ( ZeroPadding2D( (1,1) , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)
	
	
	o = (UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([ o ,f3],axis=concat_axis )  )
	o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
	o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	o = (UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([o,f2],axis=concat_axis ) )
	o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format=IMAGE_ORDERING ) )(o)
	o = ( BatchNormalization())(o)

	o = (UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([o,f1],axis=concat_axis ) )
	o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING ))(o)
	o = ( BatchNormalization())(o)


	o =  Conv2D( n_classes , (3, 3) , padding='same', data_format=IMAGE_ORDERING )( o )
	o_shape = Model(img_input , o ).output_shape
	print '======= o_shape ================================='
	print o_shape
	print o_shape[1]
	print o_shape[2]
	print '======= n_classes ================================='
	print n_classes
	outputHeight = o_shape[1]
	outputWidth = o_shape[2]

	o = (Reshape((n_classes, outputHeight*outputWidth)))(o)
	o = (Permute((2, 1)))(o)
	o = (Activation('softmax'))(o)
	model = Model( img_input , o )
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight

	print model.summary()

	return model


def VGGUnet2( n_classes ,  input_height=256, input_width=256 , vgg_level=3):
	concat_axis = 3
	assert input_height%32 == 0
	assert input_width%32 == 0

	# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
	img_input = Input(shape=(input_height,input_width,3))

	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
	f1 = x
	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
	f5 = x

	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dense( 1024 , activation='softmax', name='predictions')(x)

	vgg  = Model(  img_input , x  )
	#vgg.load_weights(VGG_Weights_path)

	levels = [f1 , f2 , f3 , f4 , f5 ]

	o = f4

	o = ( ZeroPadding2D( (1,1) , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	o = (UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([ o ,f3],axis=concat_axis )  )
	o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
	o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	o = (UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([o,f2],axis=concat_axis ) )
	o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format=IMAGE_ORDERING ) )(o)
	o = ( BatchNormalization())(o)

	o = (UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	# o = ( concatenate([o,f1],axis=concat_axis ) )
	o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING ))(o)
	o = ( BatchNormalization())(o)


	o =  Conv2D( n_classes , (3, 3) , padding='same', data_format=IMAGE_ORDERING )( o )
	o_shape = Model(img_input , o ).output_shape
	outputHeight = o_shape[1]
	outputWidth = o_shape[2]

	o = (Reshape((  n_classes , outputHeight*outputWidth   )))(o)
	o = (Permute((2, 1)))(o)
	o = (Activation('softmax'))(o)
	model = Model( img_input , o )
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight

	print model.summary()

	return model




# def VGGUnet3(n_classes,  input_height=256, input_width=256 , vgg_level=3):
#
# 	assert input_height%32 == 0
# 	assert input_width%32 == 0
#
# 	concat_axis = 3
# 	inputs = Input(shape=(input_height,input_width,3))
#
# 	#x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
# 	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')\
# 		(inputs)
# 	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
# 	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
# 	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
# 	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
# 	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
# 	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
# 	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
# 	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#
# 	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
# 	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
# 	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
# 	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
# 	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
#
# 	up_conv5 = UpSampling2D(size=(2, 2))(conv5)
# 	ch, cw = get_crop_shape(conv4, up_conv5)
# 	crop_conv4 = Cropping2D(cropping=(ch, cw))(conv4)
# 	up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
# 	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
# 	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
#
# 	up_conv6 = UpSampling2D(size=(2, 2))(conv6)
# 	ch, cw = get_crop_shape(conv3, up_conv6)
# 	crop_conv3 = Cropping2D(cropping=(ch, cw))(conv3)
# 	up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
# 	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
# 	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
#
# 	up_conv7 = UpSampling2D(size=(2, 2))(conv7)
# 	ch, cw = get_crop_shape(conv2, up_conv7)
# 	crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
# 	up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
# 	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
# 	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
#
# 	up_conv8 = UpSampling2D(size=(2, 2))(conv8)
# 	ch, cw = get_crop_shape(conv1, up_conv8)
# 	crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
# 	up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
# 	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
# 	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
#
# 	ch, cw = get_crop_shape(inputs, conv9)
# 	conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
# 	conv10 = Conv2D(n_classes, (1, 1))(conv9)
#
# 	model = Model(inputs=inputs, outputs=conv10)
#
# 	print model.summary()
#
# 	return model

def VGGUnet3(n_classes,  input_height=256, input_width=256 , vgg_level=3):
	concat_axis = 3

	inputs = Input((input_height, input_width, 3))

	conv1 = Conv2D(32, (3, 3), padding="same", name="conv1_1", activation="relu", data_format="channels_last")(inputs)
	conv1 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)
	conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool1)
	conv2 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)

	conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool2)
	conv3 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)

	conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool3)
	conv4 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)

	conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool4)
	conv5 = Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv5)

	up_conv5 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv5)
	ch, cw = get_crop_shape(conv4, up_conv5)
	crop_conv4 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv4)
	up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
	conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(up6)
	conv6 = Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv6)

	up_conv6 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv6)
	ch, cw = get_crop_shape(conv3, up_conv6)
	crop_conv3 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv3)
	up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
	conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(up7)
	conv7 = Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv7)

	up_conv7 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv7)
	ch, cw = get_crop_shape(conv2, up_conv7)
	crop_conv2 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv2)
	up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
	conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(up8)
	conv8 = Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv8)

	up_conv8 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv8)
	ch, cw = get_crop_shape(conv1, up_conv8)
	crop_conv1 = Cropping2D(cropping=(ch, cw), data_format="channels_last")(conv1)
	up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
	conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(up9)
	conv9 = Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv9)

	# ch, cw = get_crop_shape(inputs, conv9)
	# conv9  = ZeroPadding2D(padding=(ch[0],cw[0]), data_format="channels_last")(conv9)
	# conv10 = Conv2D(1, (1, 1), data_format="channels_last", activation="sigmoid")(conv9)

	flatten = Flatten()(conv9)
	Dense1 = Dense(512, activation='relu')(flatten)
	BN = BatchNormalization()(Dense1)

	#Dense2 = Dense(17, activation='sigmoid')(BN)

	model = Model(input=inputs, output=Dense2)

	return model