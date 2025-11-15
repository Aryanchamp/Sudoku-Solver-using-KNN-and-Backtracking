# Sudoku-Solver-using-KNN-and-Backtracking

This project is a **Visual Sudoku Solver** that can:
- Read a Sudoku puzzle **directly from an image**  
- Extract and recognize digits using **Computer Vision**  
- Classify digits using a **K-Nearest Neighbors (KNN)** model trained on MNIST  
- Solve the puzzle using **backtracking**  
- Show the solving process with a **clean GUI and animations**

A complete end-to-end project combining **Computer Vision**, **Machine Learning**, and **Algorithmic Problem Solving**.

---

##  Features

### 🔍 **1. Image-based Sudoku Digit Extraction**
- Detects Sudoku grid using contour detection  
- Warps perspective for perfect alignment  
- Splits into 9×9 cells  
- Performs cleanup (opening, thresholding, GaussianBlur)


### 🔢 **2. Digit Recognition (KNN Classifier)**
- Trained on the MNIST dataset  
- Preprocessing identical to training (threshold ➜ resize ➜ normalize ➜ flatten)  
- Additional validation:
  - Minimum pixel threshold  
  - Aspect ratio check  
  - Confidence threshold (if predict_proba available)

### 🧠 **3. Sudoku Solver Algorithm**
- Classic backtracking with safety checks  
- Cell-by-cell animation  
- Colored feedback:
  - Yellow → currently checking  
  - Green → number placed  
  - Red → backtracking step  

### 🖥️ **4. Tkinter GUI**
- Load Sudoku from image  
- Edit mode toggle  
- Clear board  
- Speed slider for animation  
- Highlighting of original vs solver-filled cells  

---

## 📂 Project Structure


```
Visual-Sudoku-Solver/
├── main.py                # GUI + CV + Solver
├── Knn_Train.py           # Train KNN on MNIST
├── knn_check.py           # Digit recognition test
├── KNN.sav                # Saved KNN model
├── debug_cells/           # Debug ROI images
├── sample_images/         # Pipeline images
└── README.md
```



---

# 🧠 Model Training (KNN)

The digit recognizer is a **K-Nearest Neighbors (KNN)** classifier trained on the MNIST dataset.

### 📌 Training Script (Knn_Train.py)

```python
from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize + flatten
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# Evaluate
print("Accuracy:", knn.score(x_test, y_test))

# Save model
joblib.dump(knn, "KNN.sav")
```

### ✅ Final Accuracy: **~97% on MNIST**

This model is lightweight, fast, and works well for clean digit crops.

---

# 🏞️ Image Processing Pipeline


## 🔍 Image Processing

<p align="center">
  <img src="png1.png" width="260" style="margin: 10px;"/>
  <img src="PrePro.png" width="260" style="margin: 10px;"/>
  <img src="PrePro2.png" width="260" style="margin: 10px;"/>
</p>

<p align="center">
  <img src="PrePro3.png" width="260" style="margin: 10px;"/>
  <img src="PrePro4.png" width="260" style="margin: 10px;"/>
  <img src="Single_contour.png" width="260" style="margin: 10px;"/>
</p>


---


# ▶️ How to Run
1️⃣ Install Dependencies
Run the following command to install all required libraries:
```
pip install numpy opencv-python joblib tensorflow scikit-learn
```
2️⃣ (Optional) Retrain the KNN Model

If you want to recreate the model:
```
python Knn_Train.py
```
This will train the MNIST-based KNN classifier and save:
```
KNN.sav
```
3️⃣ Run the Sudoku Solver

Just execute:
```
python main.py
```

This launches the full Tkinter GUI:

- Load Sudoku from image

- Detect digits

- Visualize solving steps

- Adjust speed

- Toggle edit mode

- Clear/reset puzzle

---

#### ⚠️ Important Notes

- Make sure KNN.sav is in the same directory as main.py

- If loading an image fails, check that it is well-lit and clear

- Debug crops are saved in debug_cells/ for troubleshooting

# 🧪 Digit Recognition Debugging

To help analyze recognition errors, every cell processed during image extraction is saved into the ``` debug_cells/``` folder.

Each file follows this naming pattern:
```
cell_{row}_{col}_pred_{digit}_conf_{value}.jpg
```
🔍 Examples:

``` cell_1_4_pred_7_conf_0.91.jpg ``` → Predicted digit 7 with 91% confidence

```cell_3_2_pred_0_conf_1.00_REASON_Too_few_pixels.jpg ``` → Marked empty due to weak ROI

```cell_5_8_pred_0_conf_1.00_REASON_Bad_aspect_ratio.jpg ```→ Invalid shape for a digit

✔ Reasons logged include:

- Low confidence prediction

- Too few non-zero pixels (weak digit)

- Bad aspect ratio (noise blob)

- Empty ROI

- No contour found

- Contour too small

These debug images are __super helpful__ when:

- Improving preprocessing

- Tweaking thresholds

- Checking why a digit was misclassified

- Planning to switch from KNN → CNN



# 🎥 Solving Demo

<p align="center"> <img src="demo.gif" width="450"/> </p>

# 📌 Future Improvements

This project already works well, but there are several high-impact upgrades that can significantly boost accuracy, speed, and real-world usability:
  - Improving character recognition 
  - Replacing KNN with CNN
  - Add Tesseract OCR
  - Deploy with streamlit
  - Mobile Camera Capture
  

