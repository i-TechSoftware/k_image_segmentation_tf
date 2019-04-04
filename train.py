import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import argparse
import Models, LoadBatches
import plotChart

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# session = tf.Session(config=config)
# K.set_session(session)

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type=str)
parser.add_argument("--train_images", type=str)
parser.add_argument("--train_annotations", type=str)
parser.add_argument("--n_classes", type=int)
parser.add_argument("--input_height", type=int, default=250)
parser.add_argument("--input_width", type=int, default=250)

parser.add_argument('--validate', action='store_false')
parser.add_argument("--val_images", type=str, default="")
parser.add_argument("--val_annotations", type=str, default="")

parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--val_batch_size", type=int, default=1)
parser.add_argument("--load_weights", type=str, default="data/weights.best.hdf5")
#parser.add_argument("--load_weights", type=str, default="data/50Epochs_224.hdf5")

parser.add_argument("--model_name", type=str, default="vgg_segnet")
parser.add_argument("--optimizer_name", type=str, default="adadelta")

args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights

optimizer_name = args.optimizer_name
model_name = args.model_name

print validate

if validate:
    val_images_path = args.val_images
    val_segs_path = args.val_annotations
    val_batch_size = args.val_batch_size

modelFns = {'vgg_segnet': Models.VGGSegnet.VGGSegnet, 'vgg_unet': Models.VGGUnet.VGGUnet,
            'vgg_unet2': Models.VGGUnet.VGGUnet2, 'vgg_unet3': Models.VGGUnet.VGGUnet3, 'fcn8': Models.FCN8.FCN8, 'fcn32': Models.FCN32.FCN32}
modelFN = modelFns[model_name]

m = modelFN(n_classes, input_height=input_height, input_width=input_width)
m.compile(loss='categorical_crossentropy', optimizer=optimizer_name, metrics=['accuracy'])

if len(load_weights) > 0:
    m.load_weights(load_weights, reshape=True)



print "Model output shape", m.output_shape

output_height = m.outputHeight
output_width = m.outputWidth

G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes,
                                           input_height, input_width, output_height, output_width)

G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes, input_height,
                                                input_width, output_height, output_width)
checkpoint = ModelCheckpoint("weights/weights.best.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = m.fit_generator(G, steps_per_epoch=367, validation_data=G2, validation_steps=101, epochs=epochs, callbacks=callbacks_list)

plotChart.PlotHistory(history)
m.save_weights(save_weights_path + ".hdf5")

# if not validate:
#     for ep in range(epochs):
#         m.fit_generator(G, 512, epochs=1)
#         m.save_weights(save_weights_path + "." + str(ep))
#         m.save(save_weights_path + ".model." + str(ep))
# else:
#     for ep in range(epochs):
#         m.fit_generator(G, 512, validation_data=G2, validation_steps=200, epochs=1)
#         m.save_weights(save_weights_path + "." + str(ep))
#         m.save(save_weights_path + ".model." + str(ep))
