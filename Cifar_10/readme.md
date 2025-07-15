CIFAR-10 Image Classification with a Custom CNN
This repository demonstrates image classification on the CIFAR-10 dataset using a custom Convolutional Neural Network (CNN) built with Keras. The CIFAR-10 dataset, a well-known benchmark in image recognition, comprises 60,000 32x32 color images across 10 distinct object classes.
Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images belonging to 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. It is split into 50,000 training images and 10,000 test images.
Model Architecture
The custom CNN architecture implemented in this project utilizes a sequential model in Keras. The model is designed to extract features from the 32x32 color images and classify them into one of the 10 categories.
The architecture is defined as follows:
python
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    # Classifier
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
Setup and Usage
Clone the repository:
bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install dependencies:
bash
pip install tensorflow keras numpy matplotlib
Run the training script (assuming your training script is named train.py):
bash
python train.py
This script will load the CIFAR-10 dataset, preprocess the data, train the CNN model, and evaluate its performance.
Results
(Include a section here to display the model's accuracy, loss, and potentially some example predictions, along with any learning curves or relevant metrics.
