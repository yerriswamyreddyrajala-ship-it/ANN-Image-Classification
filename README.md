# ANN-Based Fresh and Rotten Image Classification

## Project Overview

This project uses an Artificial Neural Network (ANN) to classify fruit and vegetable images into two categories:

- **Fresh**
- **Rotten**

The model is trained using numerical image features and predicts whether an image belongs to the Fresh or Rotten category.

## Dataset Information

- Total samples: 11,988
- Number of features: 30,000
- Problem Type: Binary Classification
- Target Classes:
  - `0` – Rotten
  - `1` – Fresh

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Workflow

1. Import required libraries.
2. Load the dataset.
3. Explore and understand the data.
4. Preprocess the numerical image features.
5. Split the dataset into training and testing sets.
6. Normalize the input features.
7. Build an ANN model.
8. Compile and train the model.
9. Apply Early Stopping to monitor validation loss.
10. Evaluate model performance.
11. Generate a classification report and confusion matrix.
12. Save the trained model.

## ANN Architecture

The model uses the following architecture:

| Layer | Configuration |
|---|---|
| Input Layer | 30,000 features |
| Hidden Layer 1 | 128 neurons, ReLU |
| Hidden Layer 2 | 64 neurons, ReLU |
| Output Layer | 1 neuron, Sigmoid |

### Model Configuration

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Evaluation Metric: Accuracy
- Epochs: 20
- Batch Size: 32
- Early Stopping: Patience = 3

## Model Evaluation

The model is evaluated using:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss
- Test Accuracy
- Test Loss
- Classification Report
- Confusion Matrix

## Project Structure

```text
ANN_Image_Classification/
│
├── ann_project_fv.ipynb
├── model_ann.keras
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd ANN_Image_Classification
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 6. Run the Notebook

Open the notebook using Jupyter Notebook or VS Code and execute the cells.

## Save and Load the Model

The trained model is saved in Keras format:

```python
model.save("model_ann.keras")
```

To load the saved model:

```python
from tensorflow.keras.models import load_model

model = load_model("model_ann.keras")
```

## Future Improvements

- Improve model performance through hyperparameter tuning.
- Experiment with different ANN architectures.
- Explore CNN-based image classification.
- Develop an image upload interface for predictions.


