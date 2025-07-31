# codtechtask3
# 🐱🐶 Cat vs Dog Image Classifier (Task 3)

This project is a binary image classification model that distinguishes between **cats and dogs** using machine learning techniques. It utilizes feature extraction with image preprocessing and classification using models like Logistic Regression or K-Nearest Neighbors.

---

## 📌 Objective

To build a supervised machine learning model that can predict whether a given image is of a **cat** or a **dog** based on pixel-level features.

---

## 🛠️ Tools & Libraries Used

- Python
- Jupyter Notebook
- `numpy`, `pandas`
- `matplotlib`, `seaborn` – for visualization
- `opencv-python` (`cv2`) – for image reading and resizing
- `sklearn` – for model training, evaluation, and data splitting

---

## 📁 Project Files


---

## 🧠 Workflow Summary

1. **Load and preprocess images**
   - Resize to uniform shape (e.g., 64x64)
   - Convert to grayscale (optional)
   - Flatten image arrays

2. **Label the data**
   - 0 for cat, 1 for dog

3. **Split data**
   - Train-test split using `train_test_split` from sklearn

4. **Train model**
   - Classifier used: `LogisticRegression`, `KNeighborsClassifier`, etc.

5. **Evaluate**
   - Accuracy, confusion matrix, classification report

---

## 🖼️ Input Data

- Images of cats and dogs stored in folders, loaded using OpenCV
- Preprocessing includes resizing and flattening for use with classical ML models

---

## 📈 Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Precision, Recall, F1-Score

---

## ▶️ How to Run

1. Clone or download this repository.
2. Place your image dataset in a `dataset/` directory with two subfolders: `cat/` and `dog/`.
3. Install required packages:
   ```bash
   pip install numpy pandas matplotlib opencv-python scikit-learn
