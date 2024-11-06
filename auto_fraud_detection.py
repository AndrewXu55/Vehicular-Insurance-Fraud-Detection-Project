# -*- coding: utf-8 -*-
"""Auto_Fraud_Detection.ipynb

This project involves creating a data pipeline and ML model to classify insurance claims as fraudulent (1) or not (0) based on images of cars.
"""

# Imports
import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import recall_score, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN

# Directory containing images
image_directory = '/path/to/image_directory'
image_train_fraud_directory = os.path.join(image_directory, 'train/Fraud')
image_train_non_fraud_directory = os.path.join(image_directory, 'train/Non-Fraud')
image_test_fraud_directory = os.path.join(image_directory, 'test/Fraud')
image_test_non_fraud_directory = os.path.join(image_directory, 'test/Non-Fraud')

# Function to get images from a directory
def get_images(directory, limit=None):
    img_list = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.jpg'):
            file_path = os.path.join(directory, file_name)
            img = Image.open(file_path)
            img_list.append(img)
            if limit and len(img_list) >= limit:
                break
    return img_list

# Load images
train_fraud_images = get_images(image_train_fraud_directory, limit=1750)
train_not_fraud_images = get_images(image_train_non_fraud_directory, limit=1750)
test_fraud_images = get_images(image_test_fraud_directory, limit=1750)
test_not_fraud_images = get_images(image_test_non_fraud_directory, limit=1750)

# Calculate mean height and width of images
def calculate_mean_dimensions(images):
    total_width, total_height = 0, 0
    for img in images:
        total_width += img.width
        total_height += img.height
    mean_width = round(total_width / len(images) / 50) * 50
    mean_height = round(total_height / len(images) / 50) * 50
    return mean_height, mean_width

mean_height, mean_width = calculate_mean_dimensions(
    test_fraud_images + train_fraud_images + test_not_fraud_images + train_not_fraud_images
)

print('Mean Height:', mean_height)
print('Mean Width:', mean_width)

# Function to resize images
def resize_images(img_list, height, width):
    resize_tuple = (width, height)
    for i in range(len(img_list)):
        img_list[i] = img_list[i].resize(resize_tuple)

# Resize images
resize_images(train_fraud_images, mean_height, mean_width)
resize_images(train_not_fraud_images, mean_height, mean_width)
resize_images(test_fraud_images, mean_height, mean_width)
resize_images(test_not_fraud_images, mean_height, mean_width)

# Function to convert images to a tensor dataset
def to_image_list(imgs):
    return [tf.keras.preprocessing.image.img_to_array(img) / 255.0 for img in imgs]

def to_dataset(imgs, label):
    tensor_imgs = to_image_list(imgs)
    labels = [label] * len(imgs)
    return tf.data.Dataset.from_tensor_slices((tensor_imgs, labels)).shuffle(buffer_size=7000)

# Create datasets
train_fraud_ds = to_dataset(train_fraud_images, 1)
train_not_fraud_ds = to_dataset(train_not_fraud_images, 0)
test_fraud_ds = to_dataset(test_fraud_images, 1)
test_not_fraud_ds = to_dataset(test_not_fraud_images, 0)

# Combine and batch datasets
batch_size = 64
train_dataset = train_fraud_ds.concatenate(train_not_fraud_ds).shuffle(buffer_size=7000).batch(batch_size)
test_dataset = test_fraud_ds.concatenate(test_not_fraud_ds).shuffle(buffer_size=7000).batch(batch_size)

# Visualize Class Proportions in Training
num_fraud = tf.data.experimental.cardinality(train_fraud_ds).numpy()
num_not_fraud = tf.data.experimental.cardinality(train_not_fraud_ds).numpy()
plt.pie(x=[num_fraud, num_not_fraud], labels=['Fraud', 'Not-Fraud'], autopct='%.2f%%')
plt.show()

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(mean_height, mean_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Accuracy()])

# Function to evaluate model performance
def evaluate_model(model, dataset):
    true_labels = []
    predictions = []
    for images, labels in dataset:
        probs = model.predict(images)
        preds = (probs > 0.5).astype(int).flatten()
        predictions.extend(preds)
        true_labels.extend(labels.numpy())
    
    recall = recall_score(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predictions)
    print(f'Recall: {recall}, Accuracy: {accuracy}')

# Evaluate base model
# model.fit(train_dataset, epochs=10, validation_data=test_dataset, verbose=1)
# evaluate_model(model, test_dataset)

# Random Sampling Functions
def randomly_undersample(non_fraud_images, ratio):
    return [img for img in non_fraud_images if random.random() <= ratio]

def randomly_oversample(fraud_images, ratio):
    new_data = fraud_images.copy()
    new_data += [img for img in fraud_images if random.random() <= ratio]
    return new_data

# Data Augmentation
def augment_data(dataset, proportion):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    num_elements = tf.data.experimental.cardinality(dataset).numpy()
    sampled_dataset = dataset.shuffle(1024).take(int(num_elements * proportion))

    def augment_image(x, y):
        x_aug = tf.numpy_function(lambda img: datagen.random_transform(img), [x], Tout=tf.float32)
        x_aug.set_shape(x.get_shape())
        return x_aug, y

    augmented_dataset = sampled_dataset.map(augment_image)
    return dataset.concatenate(augmented_dataset)

# Random Sampling Testing
undersampled_non_fraud = randomly_undersample(train_not_fraud_images, 0.75)
oversampled_fraud = randomly_oversample(train_fraud_images, 0.5)

# Create datasets for undersampled and oversampled data
undersampled_non_fraud_ds = to_dataset(undersampled_non_fraud, 0)
oversampled_fraud_ds = to_dataset(oversampled_fraud, 1)

# Create final training dataset with augmentation
augmented_resampled_train_ds = augment_data(oversampled_fraud_ds, 0.5).concatenate(undersampled_non_fraud_ds).shuffle(buffer_size=7000).batch(batch_size)

# Train on preprocessed datasets
model.fit(augmented_resampled_train_ds, epochs=10, validation_data=test_dataset, verbose=1)

# Evaluate model
recall, accuracy = evaluate_model(model, test_dataset)
print(f"Tuned Augmented & Resampled Model with Weights 1 - Recall: {recall}, Accuracy: {accuracy}")

# Save entire model to a single file
model.save('fraud_model_single_file3.h5')
