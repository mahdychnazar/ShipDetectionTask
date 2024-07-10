# Content:

1. Project deployment
2. Exploratory data analysis summary
3. Model description
4. Data preparation and input pipeline
5. Summary
 \
 \

# 1. Project deployment:

## 1.1 Setting up with conda:
1. Download the project:
```git clone <repo>```
2. Open the project in terminal:
```cd <repo>```
2. Create the environment from the environment.yml file:
```conda env create -f environment.yml```
3. Activate conda environment:
```conda activate ship_detection_env```
4. The project is ready to be tested.
``` python inference.py ./path_to_image ```

\

## 1.2 Setting up with virtual environment:
1. Download the project:
```git clone <repo>```
2. Open the project in terminal:
```cd <repo>```
3. Install virtualenv, if not installed:
```pip install virtualenv```
4. Crete new virtual environment:
```virtualenv venv```
5. Enter the virtual environment:
```source venv/bin/activate```
6. Install dependecies:
```pip install -r requirements.txt```

Versions list for running tensorflow on GPU:

python: 3.10

tensorflow: 2.10.1

CUDA: 11.2

cuDNN: 8.1
# 2. Exploratory data analysis summary:

The dataset of square images 768x768 pixels. Most of the images do not contain ships the rest contain from 1 to 15 ships. Labels are provided in .csv file, where each row contains image filename and ship position in run-length encoding format. Overall, there are 81723 ships on 192556 images. All images have same size. 

# 3. Model description

The model implements U-net architecture for neural network. (<https://arxiv.org/abs/1505.04597>).

The model does not reproduce the original U-net architecture, but it implements the general concept of it.

The input layer of the model takes tensor made of 768x768x3 image.

The output layer produces tensor of shape 768x768x1. The purpose is to produce the grayscale mask of ships on an image. The sigmoid activation function is used for binary (ship/background) classification of pixels of the mask.

Encoder blocks are sequences of 3x3 convolutional layers, 2x2 max-pooling layers, batch-normalization and ReLU activation function.

Decoder blocks are sequences of 2x2 transpose convolutional layers with stride 2, batch-normalization, ReLU activation function. Some decoder blocks produce dropout with probability (0.25) to prevent overfitting.

There are skip connections between encoder and decoder blocks where decoder block input is (2x, y, z, w), where (x, y, z, w) is encoder block output. In other words, decoder block input and encoder block output are stack together.

Learning policy: The model weights are initialized using Glorot method. The learning rate starts from 0.01. If the validation loss does not improve after one epoch of training, the learning rate is reduced ten times. If the validation loss does not improve after 3 epochs of training the training is stopped. The final values of weights are also saved in file.

The model graph is given in “modelgraph” folder, filename: “model\_grapn.png”

# 4. Data preparation and input pipeline

The labels are in run-length encoding format, so they are decoded and saved on disk as .jpg images containing grayscale mask of ships. The mask name matches the original image name, so in order to make the training dataset it is necessary to get full list of training images names from label file.

Most of images does not contain ships so the number of background-only images is reduced to prevent bias.

Validation set: 3000 images.

Train set the rest of images.

Data augmentation: Each image is flipped horizontally, vertically, or both horizontally and vertically. According to the fact that the satellite shots are made top-down, this kind of augmentation does not affect the data relevance, but increases the overall amount and variety of data.

Batch size: 8 images.

# 5. Summary:

Kaggle private score: 0.75142

![](Aspose.Words.ca97ab67-cb41-41cb-b590-cd045f289e76.001.png)
