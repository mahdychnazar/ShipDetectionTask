import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image


#ref https://www.kaggle.com/code/kmader/baseline-u-net-model-part-1?scriptVersionId=6617956&cellId=5
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)



#ref https://www.kaggle.com/code/kmader/baseline-u-net-model-part-1?scriptVersionId=6617956&cellId=5
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


#Returns mask with all labeled ships on an image
def mask_decode(rle_rows, shape):
    #print(rle_rows)
    if len(rle_rows) == 0:
        return np.zeros(shape, dtype=np.uint8)

    all_ships = np.zeros(shape, dtype=np.uint8)
    for rle_code in rle_rows['EncodedPixels']:
        if isinstance(rle_code, str):
            all_ships += rle_decode(rle_code, shape)
    return all_ships.T

#save string summary into 'modelsummary.txt' file
def myprint(s):
    if not os.path.exists('./modelsummary'):
        # Create the directory
        os.makedirs('./modelsummary')
        print("Directory created successfully!")
    with open('./modelsummary/modelsummary.txt','a') as f:
        print(s, file=f)

#Saves all the lasbels on disk as .jpg masks
def save_all_labels(metadata):
    if not os.path.exists('.//data//airbus-ship-detection//train_label_v2//'):
        # Create the directory
        os.makedirs('.//data//airbus-ship-detection//train_label_v2//')
        print("Directory created successfully!")
        for image_name in metadata['ImageId'].unique():
            rle_rows = metadata[metadata['ImageId'] == image_name]
            mask = mask_decode(rle_rows, (768, 768))
            cv2.imwrite(os.path.join('.//data//airbus-ship-detection//train_label_v2//', image_name), mask)
        return
    else:
        print("Directory already exist!")
        return

def load_image(image):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image)
    image = tf.image.resize(image, (768, 768))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.0
    image = tf.reshape(image, [1, 768, 768, 3])

    return image
