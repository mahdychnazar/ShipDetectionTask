import numpy as np
import model as md
import utils as ut
import tensorflow as tf
import cv2
import os
import sys

#python inference.py examples\00dc34840.jpg
#python inference.py examples

#makes the model output stackable with the original image
def make_prediction(image_path):
    image = ut.load_image(image_path)
    predict =md.small_unet.predict(image)
    predict = predict.reshape((768,768) + (1,))
    predict = np.array(predict)
    predict = np.squeeze(predict)
    predict = predict.astype(np.uint8)
    predict = predict * 255
    predict = np.dstack([predict, predict, predict])
    return predict


if __name__ == '__main__':
    input = sys.argv[1]
    output_path = ".//output//"

    if not os.path.exists(output_path):
        os.mkdir(output_path) 

    md.small_unet.load_weights('./model/best_small_unet.h5')
    
    if os.path.exists(input):
        if os.path.isfile(input):
            #if input argument is an image file, the model output is displayed and saved into "./output" folder.
            if input.endswith(".jpeg") or input.endswith(".jpg"):
                image = cv2.imread(input)
                image = cv2.resize(image, (768,768))  
                predict = make_prediction(input)
                toshow = np.concatenate((image , predict), axis = 1)
                cv2.imwrite(os.path.join(output_path, os.path.basename(input).split('/')[-1]), predict)
                cv2.imshow(os.path.basename(input).split('/')[-1], toshow)
                cv2.waitKey(0)
        else:
            #if input argument is a folder path, all the images inside are taken as input
            #and the model output is saved into "./output" folder.
            for file in os.listdir(input):
                if(file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".jpg")):
                    file = os.path.join(input, file)
                    image = cv2.imread(input)
                    image = cv2.resize(image, (768,768))
                    predict = make_prediction(file)
                    cv2.imwrite(os.path.join(output_path, os.path.basename(file).split('/')[-1]), predict)
                
    else:
        print("Path does not exist")
        