# Vehicular Insurance Fraud-Detection Project

## Functionality

- **Image Upload**: Users can upload images through the web interface.
- **Image Preprocessing**: The uploaded image is resized and normalized for prediction.
- **Model Prediction**: The pre-trained CNN model processes the image to classify it as either "Fraud!" or "Not Fraud!".
- **Result Display**: The prediction result and probability are displayed on the same page after submission.

## Files and Functions

- **app.py**: The main application file that handles web requests, image processing, and model predictions. It sets up routes for the home page and prediction functionality.
- **auto_fraud_detection.py**: A module dedicated to automated fraud detection. It integrates model loading and prediction functionalities and can be extended for batch processing or real-time fraud detection scenarios. This file can serve as a standalone script for running predictions on multiple images or files without the web interface.
- **model_splitter.py**: A module for splitting large HDF5 model files into smaller parts and combining them when necessary.
- **index.html**: The HTML template for the web interface, where users can upload files and view predictions.
- **styles.css**: A stylesheet for styling the web application interface.
- **requirements.txt**: A file listing all required Python libraries and their versions.

## Contributors
 - [Samuel Kinstlinger](https://deepmind.google/about/student-researcher-program/)
 - Rithvik Bonagiri
 - Andrew Xu
 - Bari Vadaria
