import os
import time
import argparse

import cv2
import numpy as np
from keras.models import load_model
from keras.utils import CustomObjectScope
from scipy.misc import imread, imresize, imsave

from utils.custom_objects import custom_objects
from utils.loss import np_dice_coef
from nets.DeeplabV3plus import DeeplabV3plus
from nets.Prisma import PrismaNet

def blend_img_with_mask(img, alpha, img_shape):
    mask = alpha >= 0.99
    mask_3 = np.zeros(img_shape, dtype='float32')
    mask_3[:,:,0] = 255
    mask_3[:,:,0] *= alpha
    result = img*0.5 + mask_3*0.5
    return np.clip(np.uint8(result), 0, 255)

def evaluate(model_path, imgs_path, input_shape, out_path):
    with CustomObjectScope(custom_objects()):
        model = load_model(model_path)
        # model.summary()
    
    imgs = [f for f in os.listdir(imgs_path)]
    for _ in imgs:
        img = imread(os.path.join(imgs_path, _), mode='RGB')
        img_shape = img.shape
        input_data = img.astype('float32')
        inp_data = input_data
        input_data = imresize(img, input_shape)
        input_data = input_data / 255.
        input_data = (input_data - input_data.mean()) / input_data.std()
        input_data = np.expand_dims(input_data, axis=0)
        
        output = model.predict(input_data)

        mask = cv2.resize(output[0,:,:,0], (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = np.array(mask, dtype='uint8')
        inp_data = np.array(inp_data, dtype='uint8')
        print(mask.shape," ",inp_data.shape)
        res = cv2.bitwise_and(inp_data,inp_data,mask = mask)
        
        # img_with_mask = blend_img_with_mask(img, mask, img_shape)
        imsave(out_path + _, res)


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='./models/CelebA_DeeplabV3plus_256_hair_seg_model.h5')
    parser.add_argument('--imgs_path', default='./imgs/test')
    parser.add_argument('--input_shape', default=[256,256])
    parser.add_argument('--out_path', default="./out")
    args = parser.parse_args()
    evaluate(args.model_path, args.imgs_path, args.input_shape, args.out_path)


if __name__ == "__main__":
    main()



