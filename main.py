# Install Dependencies:
!pip install tensorflow keras numpy pandas matplotlib opencv-python scikit-learn


#Import Libraries:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os



#Importing Data from Kaggle Data Sets:
import kagglehub
# Download latest version
path = kagglehub.dataset_download("emmarex/plantdisease")
print("Path to dataset files:", path)

#Splitting data to train and test:
train_dir = "/kaggle/input/plantdisease/train"
test_dir  = "/kaggle/input/plantdisease/test"

#Training data to categories:
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)


#Build CNN Model:
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#Train the Model:
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10,
    verbose=1
)

#Evaluate and Visualize:
plt.figure(figsize=(8,4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

#Save Model:
model.save("plant_disease_model.h5")
print("✅ Model saved successfully!")

#Test with a New Leaf Image:
from google.colab import files
from tensorflow.keras.preprocessing import image
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0
pred = model.predict(img_array)
predicted_class = np.argmax(pred)
class_labels = list(train_data.class_indices.keys())
print("🌿 Predicted Disease Type:", class_labels[predicted_class])


