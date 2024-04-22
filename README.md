# Senior Capstone
Repository for CSU Fullerton Computer Science Senior Capstone Project - Emotion Detection App

## Synthetic Dataset
* Copy code from Synthetic_Dataset_Creation into Google Colab/Jupyter Notebook to start building Synthetic Dataset. Alternatively, download a dataset online that already has folders of people showing emotions of specified types.

## Prepare Data
* Manually cleanse the data of abnormalities to ensure a positive training model (> 70%)

## Run prepare_data.py
* This will build the model data by extracting landmarks from the data set

## Run train_model.py
* This will train the model based off of your data set. Always shoot for above 70% but the higher the better. A lower score means the data was not clear for the model to decipher. Use the Confusion Matrix to determine which emotions need adjusting.

## Run test_model.py
* This will open the camera in a window and allow for you to test how well the app is reading the emotions.
