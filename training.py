import pandas as pd
import numpy as np

import os
import tensorflow as tf
import opendatasets as od
import utils as ut

import model


CLASSES = 2

SAMPLE_SIZE = (768, 768)

OUTPUT_SIZE = (768, 768)

#Download data is necessary:

od.download("https://www.kaggle.com/c/airbus-ship-detection/data", data_dir='data')

#Load labels:
metadata = pd.read_csv(".//data//airbus-ship-detection//train_ship_segmentations_v2.csv")

#Save labels if necessary
ut.save_all_labels(metadata)

im_path = ".//data//airbus-ship-detection//train_v2//"
ms_path = ".//data//airbus-ship-detection//train_label_v2//"

model_file = 'small_unet.h5'
log_file = 'log.csv'

#Load image as (768X768X3) tensor, load rle label as 768x768x1 tensor
def load_image_mask(image, mask):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, OUTPUT_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.0
    image = tf.reshape(image, [768, 768, 3])

    #decoding rle masks
    mask = tf.io.read_file(mask)
    mask = tf.io.decode_png(mask)
    mask = tf.image.resize(mask, OUTPUT_SIZE)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.reshape(mask, [768, 768, 1])
    return image, mask

#Extending the dataset
def augmentate_image_mask(image, mask):
    flip_prob = tf.random.uniform((), 0, 1)

    #probability 0.7 to flip horizontally
    #probability 0.7 to flip vertically
    #probability 0.4 to flip both

    if flip_prob <= 0.7:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if flip_prob >= 0.3:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    image = tf.image.resize(image, SAMPLE_SIZE)
    mask = tf.image.resize(mask, SAMPLE_SIZE)

    return image, mask


#Training-validation set
def split():
    # Reduce the amount of images containing background only
    # for faster training and avoiding bias
    empty_images_num = 5000

    empty_ims = metadata[metadata['EncodedPixels'].isna()]['ImageId'].values[:empty_images_num]
    nonempty_ims = metadata[metadata['EncodedPixels'].notna()]['ImageId'].unique()
    images_list = np.append(empty_ims, nonempty_ims)
    np.random.shuffle(images_list)

    train_images_list, val_images_list = images_list[3000:], images_list[:3000]

    print(train_images_list.shape)
    print(val_images_list.shape)

    train_mask_list = ms_path + train_images_list
    train_images_list = im_path + train_images_list

    val_mask_list = ms_path + val_images_list
    val_images_list = im_path + val_images_list

    return train_images_list, train_mask_list, val_images_list, val_mask_list

#Create data pipeline
def tf_dataset(ims, msks,  batch = 8):
    dataset = tf.data.Dataset.from_tensor_slices((ims,msks))
    dataset = dataset.map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
    aug_dataset = dataset.map(augmentate_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.concatenate(aug_dataset)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def run_training():
    train_images_list, train_mask_list, val_images_list, val_mask_list = split()
    train_dataset = tf_dataset(train_images_list, train_mask_list)
    val_dataset = tf_dataset(val_images_list, val_mask_list)
    print(train_dataset)
    model.small_unet_train(train_dataset, val_dataset, 20)


run_training()
