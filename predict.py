import argparse
import Models, LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import random
import os
from keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument("--load_weights_path", type=str)
parser.add_argument("--epoch_number", type=int, default=5)
parser.add_argument("--test_images", type=str, default="")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--input_height", type=int, default=224)
parser.add_argument("--input_width", type=int, default=224)
parser.add_argument("--model_name", type=str, default="vgg_segnet")
parser.add_argument("--n_classes", type=int)
parser.add_argument("--one_file_path", type=str, default="")

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
input_width = args.input_width
input_height = args.input_height
output_path = args.output_path
epoch_number = args.epoch_number
one_file_path = args.one_file_path

modelFns = {'vgg_segnet': Models.VGGSegnet.VGGSegnet, 'vgg_unet': Models.VGGUnet.VGGUnet,
            'vgg_unet2': Models.VGGUnet.VGGUnet2, 'fcn8': Models.FCN8.FCN8, 'fcn32': Models.FCN32.FCN32}
modelFN = modelFns[model_name]

with K.tf.device('/cpu:0'):
    m = modelFN(n_classes, input_height=input_height, input_width=input_width)
    m.load_weights(args.load_weights_path)
    m.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

output_height = m.outputHeight
output_width = m.outputWidth


if one_file_path!='':
    images = one_file_path
else:
    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]

def GetColorClass(n_class):
    """ store label data to colored image """
    Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Pole = [192, 192, 128]
    Road = [128, 64, 128]
    Pavement = [60, 40, 222]
    Tree = [128, 128, 0]
    SignSymbol = [192, 128, 128]
    Fence = [64, 64, 128]
    Car = [64, 0, 128]
    Pedestrian = [64, 64, 0]
    Bicyclist = [0, 128, 192]
    Unlabelled = [0, 0, 0]
    label_colours = np.array(
        [Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist,
         Unlabelled])
    return label_colours[n_class]


def PredictAndSave(imgName):
    print 'Predecate file: '+imgName
    path, filename = os.path.split(imgName)
    outName = os.path.join(args.output_path, filename)
        #imgName.replace(images_path, args.output_path)
    X, orig_height, orig_width = LoadBatches.getImageArr(imgName, args.input_width, args.input_height)
    with K.tf.device('/cpu:0'):
        pr = m.predict(np.array([X]))[0]
        pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
        seg_img = np.zeros((output_height, output_width, 3))
        for c in range(n_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (GetColorClass(c)[0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (GetColorClass(c)[1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (GetColorClass(c)[2])).astype('uint8')

    # cv2.imshow('Segmental image', seg_img)
    # cv2.waitKey()
    seg_img = cv2.resize(seg_img, (orig_width, orig_height))

    print 'Save file to:' + outName

    cv2.imwrite(outName, seg_img)
    cv2.imshow('Predict', seg_img)
    cv2.waitKey()
    return 0


if one_file_path!='':
    PredictAndSave(one_file_path)
else:
    for imgName in images:
        PredictAndSave(imgName)


