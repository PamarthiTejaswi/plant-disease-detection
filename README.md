# 🌿 Plant Disease Detection using Deep Learning (CNN)

This project aims to **detect and classify plant leaf diseases** using **Convolutional Neural Networks (CNNs)** trained on the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).  
By analyzing leaf images, the model can predict whether a plant is **healthy** or affected by a **specific disease**, helping farmers and researchers identify crop problems early.

---

## 📸 Dataset

**Source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

- Contains over **50,000+ images** of **plant leaves** across multiple crop types and diseases.
- Each image is categorized into one of several classes (e.g., *Tomato___Early_blight*, *Tomato___Healthy*).

**Structure after preprocessing:**


---

## 🧠 Project Workflow

### **1️⃣ Data Preparation**
- Downloaded dataset from Kaggle using `kagglehub`.
- Split dataset into **train (80%)** and **validation (20%)** using `splitfolders`.
- Preprocessed and augmented images using `ImageDataGenerator` in Keras.

### **2️⃣ Model Building**
A **Convolutional Neural Network (CNN)** architecture was implemented using **TensorFlow/Keras**:
- **Conv2D + MaxPooling** layers for feature extraction.  
- **Flatten + Dense** layers for classification.  
- **Dropout** and **L2 regularization** for preventing overfitting.

### **3️⃣ Model Training**
- Optimizer: `Adam`
- Loss Function: `categorical_crossentropy`
- Epochs: 10–20 (depending on accuracy)
- Achieved **~95% validation accuracy**

### **4️⃣ Evaluation**
- Plotted training and validation accuracy/loss.
- Evaluated model on unseen images.
- Exported the trained model as `plant_disease_model.h5`.

### **5️⃣ Prediction**
- Uploaded a leaf image.
- Preprocessed and passed it through the CNN.
- Output: **Predicted Disease Name**

---

## 🧩 Model Architecture
| Layer | Output Shape | Parameters |
|--------|----------------|-------------|
| Conv2D (32 filters) | (126,126,32) | 896 |
| MaxPooling2D | (63,63,32) | 0 |
| Conv2D (64 filters) | (61,61,64) | 18496 |
| MaxPooling2D | (30,30,64) | 0 |
| Flatten | (57600) | 0 |
| Dense (128 neurons, ReLU) | (128) | 7372928 |
| Dropout (0.5) | - | - |
| Dense (Softmax Output) | (#Classes) | - |

---

## ⚙️ Tools & Technologies

| Category | Tools Used |
|-----------|------------|
| **Programming Language** | Python |
| **Deep Learning Framework** | TensorFlow, Keras |
| **Data Preprocessing** | OpenCV, NumPy, Pandas |
| **Augmentation** | ImageDataGenerator |
| **Visualization** | Matplotlib |
| **Dataset Source** | Kaggle (PlantVillage) |
| **Environment** | Google Colab (GPU-enabled) |

---

## 📊 Results

| Metric | Result |
|--------|--------|
| Training Accuracy | ~97% |
| Validation Accuracy | ~94% |
| Test Accuracy | ~92% |
| Output Example | `Predicted: Tomato___Early_blight` |

**Accuracy vs Epochs:**
![Accuracy Graph](https://user-images.githubusercontent.com/example/accuracy.png)

---

## 🌱 Example Prediction

```bash
🌿 Predicted Disease: Tomato___Early_blight

How to Run This Project
🔧 Step 1: Clone the Repository
git clone https://github.com/<your-username>/Plant-Disease-Detection.git
cd Plant-Disease-Detection

🧩 Step 2: Install Dependencies
pip install tensorflow keras numpy pandas matplotlib opencv-python scikit-learn split-folders kagglehub

🌿 Step 3: Run the Notebook
Open the Jupyter/Colab notebook and execute all cells in order:
Plant_Disease_Detection.ipynb

💾 Model Saving & Deployment
Saved model: plant_disease_model.h5
Can be integrated into:
Web apps (Flask/Django)
Mobile apps (TensorFlow Lite)
IoT devices for smart farming

📚 Future Enhancements
🌾 Use Transfer Learning (MobileNetV2, ResNet50) for higher accuracy.
🧠 Develop a web interface for real-time detection.
📱 Build a mobile app for farmers to upload leaf photos.
☁️ Deploy on Streamlit or Flask for live demos.

⭐ If you found this project useful, don’t forget to star the repo!

---

Would you like me to generate an **“Abstract + Objective + Conclusion”** section in Word-style format too (for your report or README bottom)?

