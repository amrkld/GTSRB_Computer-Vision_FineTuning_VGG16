# German Traffic Sign Recognition with VGG16

This repository contains a fine-tuning of the VGG16 pre-trained model to solve a classification problem using images of German traffic signs. The goal is to predict the type of traffic sign in the images. This project is part of a competition hosted on Kaggle.

**Important note:** You cannot submit the code to the Kaggle competition without first modifying the prediction output.

## Dataset

The dataset used for this project can be downloaded from Kaggle. @ [Kaggle Competition: GTRSR - German Traffic Sign Recognition](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

The dataset consists of:
- **Training set:** Images and labels for training the model.
- **Validation set:** Images for evaluating the model.
- **Classes:** 43 different classes of stop signs

- `Data/`: Contains the training images and their respective CSV files.
- `GTSRB_Classification_Using_VGG.ipynb`: Jupyter notebook containing the entire workflow from data preprocessing to model evaluation.

## Required Libraries

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib`: For data visualization.
- `tensorflow` and `keras`: For building and training the CNN model.

## Usage

1. **Download the dataset from Kaggle.**

2. **Open the Jupyter notebook `GTSRB_Classification_Using_VGG.ipynb` and run all the cells to execute the following steps:**

    - Load and preprocess the data.
    - Visualize the data distribution.
    - Build and compile the CNN model.
    - Train the model.
    - Evaluate the model.
    - Make predictions on the test set.

## Model Architecture

The CNN model is built using TensorFlow and Keras. The architecture consists of:
- **Base Model:** VGG16 pre-trained on ImageNet.
- **Custom Layers:** 
  - Global Average Pooling layer.
  - Dense layer with softmax activation for the output layer to classify the traffic signs.

## Results

The model is trained for 20 epochs (as a start) with data augmentation.
The training and validation accuracy and loss are plotted to visualize the model's performance.

## Important note

Model accuracy isn't good after training it for 20 epochs because of the complex model architecture. So, we have two options.
- First is to increase the number of the epochs and change slightly in the model (adding drop out to avoid overfitting) -> will be expenses
- We can choose another model like MobileNetV2 or EfficientNet-B0 -> we will try this soon 

## Acknowledgements

Special thanks to Kaggle for providing the dataset and hosting the competition. This project is an excellent opportunity to apply deep learning techniques to real-world image classification tasks.

[Kaggle Competition: GTRSR - German Traffic Sign Recognition](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
