import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from keras.utils import plot_model

import zipfile
from PIL import Image

warnings.filterwarnings('ignore')

# Streamlit configurations
#st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Gender Prediction from UTKFace Dataset")

# Function to extract the zip file
def extract_zip(zip_file_path, extract_to_path):
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
    except zipfile.BadZipFile:
        st.error("Error: The file is not a zip file or it is corrupted.")

# Function to read images and labels from the specified folder
def read_images_and_labels_from_folder(folder_path, num_images=1500):
    image_paths = []
    age_labels = []
    gender_labels = []
    ethnicity_labels = []
    images = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                file_path = os.path.join(root, file)
                try:
                    image = Image.open(file_path)
                    images.append(image)
                    
                    temp = file.split('_')
                    age = int(temp[0])
                    gender = int(temp[1])
                    ethnicity = int(temp[2])
                    
                    image_paths.append(file_path)
                    age_labels.append(age)
                    gender_labels.append(gender)
                    ethnicity_labels.append(ethnicity)
                    
                    if len(images) >= num_images:
                        break
                except Exception as e:
                    st.error(f"Error processing file {file_path}: {e}")
        if len(images) >= num_images:
            break
    
    return images, image_paths, age_labels, gender_labels, ethnicity_labels

# Function to extract features from images
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode='grayscale')
        img = img.resize((128, 128), Image.LANCZOS)
        img = np.array(img)
        features.append(img)
        
    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)
    return features

# File uploader
uploaded_file = st.file_uploader("Choose a zip file containing the UTKFace dataset", type="zip")

if uploaded_file is not None:
    with st.spinner('Extracting zip file...'):
        extract_to_path = 'extracted'
        os.makedirs(extract_to_path, exist_ok=True)
        extract_zip(uploaded_file, extract_to_path)
        st.success('Zip file extracted successfully')

    utkface_folder_path = os.path.join(extract_to_path, 'UTKFace')
    if os.path.exists(utkface_folder_path):
        with st.spinner('Reading images and labels...'):
            images, image_paths, age_labels, gender_labels, ethnicity_labels = read_images_and_labels_from_folder(utkface_folder_path)
            st.success('Images and labels read successfully')

        df = pd.DataFrame()
        df['image'], df['gender'] = image_paths, gender_labels
        gender_dict = {0: 'Male', 1: 'Female'}

        if st.checkbox('Show Image Grid'):
            plt.figure(figsize=(20, 20))
            files = df.iloc[0:25]
            for index, file, gender in files.itertuples():
                plt.subplot(5, 5, index + 1)
                img = load_img(file)
                img = np.array(img)
                plt.imshow(img)
                plt.title(f" Gender: {gender_dict[gender]}")
                plt.axis('off')
            st.pyplot(plt)

        if st.checkbox('Show Gender Distribution'):
            sns.countplot(df['gender'])
            st.pyplot()

        with st.spinner('Extracting features...'):
            X = extract_features(df['image'])
            X = X / 255.0
            y_gender = np.array(df['gender'])
            st.success('Features extracted successfully')

        input_shape = (128, 128, 1)
        inputs = Input((input_shape))
        conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
        maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
        conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxp_1)
        maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
        conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(maxp_2)
        maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
        conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu')(maxp_3)
        maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

        flatten = Flatten()(maxp_4)
        dense_1 = Dense(256, activation='relu')(flatten)
        dropout_1 = Dropout(0.4)(dense_1)
        output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_1)

        model = Model(inputs=[inputs], outputs=[output_1])
        model.compile(loss=['binary_crossentropy'], optimizer='adam', metrics=['accuracy'])

        if st.checkbox('Train Model'):
            with st.spinner('Training model...'):
                history = model.fit(x=X, y=[y_gender], batch_size=32, epochs=30, validation_split=0.2)
                st.success('Model trained successfully')

            acc = history.history['gender_out_accuracy']
            val_acc = history.history['val_gender_out_accuracy']
            epochs = range(len(acc))

            plt.plot(epochs, acc, 'b', label='Training Accuracy')
            plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
            plt.title('Accuracy Graph')
            plt.legend()
            st.pyplot()

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            plt.plot(epochs, loss, 'b', label='Training Loss')
            plt.plot(epochs, val_loss, 'r', label='Validation Loss')
            plt.title('Loss Graph')
            plt.legend()
            st.pyplot()

        image_index = st.slider('Select Image Index for Prediction', 0, len(df)-1, 0)
        if st.button('Predict'):
            st.write("Original Gender:", gender_dict[y_gender[image_index]])
            pred = model.predict(X[image_index].reshape(1, 128, 128, 1))
            pred_gender = gender_dict[round(pred[0][0])]
            st.write("Predicted Gender:", pred_gender)
            plt.axis('off')
            plt.imshow(X[image_index].reshape(128, 128), cmap='gray')
            st.pyplot()
