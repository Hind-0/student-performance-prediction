🎓 Student Pass/Fail Prediction Using Machine Learning

A machine learning project investigating whether student pass/fail outcomes can be predicted using student-related information, and comparing the performance of Decision Tree and Multi-Layer Perceptron (MLP) models.

📌 Project Overview

This project uses machine learning to classify students into two categories:

Pass
Fail

The project investigates how different types of student information affect prediction performance.

Three different feature sets are investigated:

Model A: Non-academic student features
Model B: Non-academic features + first-semester grade (G1)
Model C: Non-academic features + first-semester grade (G1) + second-semester grade (G2)

The pass/fail target is created using the final grade (G3), where:

G3 ≥ 8 = Pass
G3 < 8 = Fail
🔬 Research Question

Can machine learning models accurately predict whether a student will pass or fail, and how does including previous academic grades affect prediction performance?

💡 Hypothesis

Machine learning can be used to predict student pass/fail outcomes, and including previous semester grades will improve the predictive performance of the models.

🤖 Models Used
Decision Tree

Three Decision Tree classifiers are trained using different feature sets.

Decision Tree A: Non-academic features
Decision Tree B: Non-academic features + G1
Decision Tree C: Non-academic features + G1 + G2

The models are evaluated using different maximum tree depths.

Multi-Layer Perceptron (MLP)

Three neural network models are also trained using the same feature sets as the Decision Trees.

The MLP architecture consists of:

Input Features
      ↓
Dense Layer — 50 neurons
      ↓
Dense Layer — 20 neurons
      ↓
Sigmoid Output

The MLP models use Min-Max scaling, ReLU activation functions, binary cross-entropy loss, and the Adam optimizer.

🔢 Data Processing

The dataset is processed using Python libraries including Pandas and Scikit-learn.

The preprocessing steps include:

Checking for missing values
Converting categorical variables into numerical features using one-hot encoding
Creating a binary Pass/Fail target
Splitting the data into 80% training and 20% testing sets
Applying Min-Max scaling to the MLP input features

📊 Evaluation

The models are evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion matrices
Predicted versus actual pass rates

Decision Tree visualisations and MLP training graphs are also produced to analyse model behaviour.

🔎 Research Areas

The project investigates:

How accurately can student outcomes be predicted?
Which student features contribute to prediction?
Does including G1 improve prediction performance?
Does including G2 further improve prediction performance?
How do Decision Trees compare with MLP neural networks?
How do the models differ in their predicted pass rates?
Which model provides the most useful classification results?

🗂️ Repository Structure
student-performance-prediction/

├── data/

│   └── Student_Math_Data.csv

│
├── results/

│   ├── tree_A.png

│   ├── tree_B.png

│   ├── tree_C.png

│   ├── Model_A.png

│   ├── Model_B.png

│   └── Model_C.png

│
├── student_prediction.py

├── requirements.txt

└── README.md

🛠️ Technologies

-Python

-Pandas

-NumPy

-Scikit-learn

-TensorFlow / Keras

-Matplotlib

👩‍💻 Author

Hind Michaal

Computer Science Student/
Machine Learning/
Interested in Cyber Security
