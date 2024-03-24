# Emoji and Facial Expression Classification Project
This repository contains the code and methodology for a computer vision project that focuses on the classification of emojis and extends its approach to facial expression recognition using machine learning models.

## Dataset Preparation
### Emoji Dataset
The emoji dataset was constructed by collecting emoji images and categorizing them into distinct groups. These images were used to train and test various machine learning classifiers.

### Facial Expression Recognition (FER) Dataset
To replicate the facial expression recognition part of this project, you will need to download the FER2013 dataset from the Kaggle competition titled "Challenges in Representation Learning: Facial Expression Recognition Challenge". Follow these steps:

+ Go to Kaggle competition page.
+ Download the dataset to your local machine.
+ Create a folder in your project directory named fer_data.
```
mkdir fer_data
```
+ Extract the downloaded dataset into the fer_data folder.
```
unzip -d ./fer_data challenges-in-representation-learning-facial-expression-recognition-challenge.zip
```

## Installation
Before running the project, you must install the necessary Python packages. A list of required packages is provided in the requirements.txt file in this repository.

### Required Packages

You may install the required packages using pip. Run the following command in your terminal:
```
pip install -r requirements.txt
```
