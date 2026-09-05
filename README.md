Predicting Student Pass/Fail Outcomes Using Decision Trees and Multi-Layer Perceptrons
Project Overview

This project uses machine learning to predict whether a student will pass or fail based on student-related features.

Two machine learning approaches are compared:

Decision Tree classifiers
Multi-Layer Perceptron (MLP) neural networks

Three models are developed:

Model A: Non-academic student features
Model B: Non-academic features + first-semester grade (G1)
Model C: Non-academic features + first-semester grade (G1) + second-semester grade (G2)

The target variable is binary:

0 = Fail
1 = Pass
Technologies Used
Python
Pandas
NumPy
Scikit-learn
TensorFlow / Keras
Matplotlib
Machine Learning Methods

The Decision Tree models are trained using different maximum depths.

The MLP models use:

Min-Max scaling
ReLU activation functions
Sigmoid output
Binary cross-entropy loss
Adam optimizer
Evaluation

The models are evaluated using:

Accuracy
Classification report
Confusion matrices
Predicted versus actual pass rates
Dataset

The project uses the Student_Math_Data.csv dataset.

How to Run

Install the required Python libraries and run:

python student_prediction.py
