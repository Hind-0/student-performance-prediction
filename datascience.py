import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing._data import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay

#/// Task 1 with standardisation included in MLPs (task3) ///

#reads the csv file and prints it
data = pd.read_csv("Student_Math_Data.csv")
pd.set_option('display.max_columns', None)
print(data.head())
print(data.info())
print()


#checks for missing values 
print('Missing values checker')
print(data.isnull().sum())

#convert categorical colums into numbers 
data=pd.get_dummies(data)

#binary pass grade
data['Passgrade']=(data['G3']>=8).astype(int) #40% of 20 is 8

x=data.drop(['G3','Passgrade'],axis=1)
y = data['Passgrade']


#/// task 2 and 3 ///
#plotting conufusion matrix
def plot_cm(y_true, y_pred, title):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Fail", "Pass"],
        cmap="Blues"
    )
    disp.ax_.set_title(title)
    plt.savefig(title)
    plt.show()


# /// Task 2 ///

#non academic features in tree A
tree_A = ['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
       'freetime', 'health', 'absences', 'school_GP', 'school_MS','address_R',
       'address_U', 'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T',
       'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services',
       'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_other',
       'Fjob_services', 'Fjob_teacher', 'reason_course', 'reason_home',
       'reason_other', 'reason_reputation', 'guardian_father',
       'guardian_mother', 'guardian_other', 'schoolsup_no', 'schoolsup_yes',
       'famsup_no', 'famsup_yes', 'paid_no', 'paid_yes', 'activities_no',
       'activities_yes', 'higher_no',
       'higher_yes', 'internet_no', 'internet_yes']   

#features in tree B including first semester grades
tree_B =tree_A + ['G1'] 

#features in tree C including second semester grades
tree_C =tree_B +['G2']  

#dataset split into 80% training and 20% testing 
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

X_A = data[tree_A]
X_B = data[tree_B]
X_C = data[tree_C]

X_A_train=X_train[tree_A]
X_A_test=X_test[tree_A]

X_B_train=X_train[tree_B]
X_B_test=X_test[tree_B]

X_C_train=X_train[tree_C]
X_C_test=X_test[tree_C]


#Tree A
dt_A = tree.DecisionTreeClassifier(random_state=42, max_depth=5)
dt_A.fit(X_A_train,y_train)
y_pred_A = dt_A.predict(X_A_test)
 
#model performance evaluation used in report
print("Accuracy:", accuracy_score(y_test, y_pred_A))
print(classification_report(y_test, y_pred_A))

plot_cm(y_test, y_pred_A, "Decision Tree A – Confusion Matrix")

fig= plt.figure(figsize=(15,12))
tree.plot_tree(dt_A,
            filled=True,
            feature_names=X_A.columns,
             class_names=['Fail','Pass'] )
plt.title('Decision Tree A')
plt.savefig("tree_A.png") #picture used in report
plt.show()


#Tree B 
dt_B = tree.DecisionTreeClassifier(random_state=42, max_depth=10)
dt_B.fit(X_B_train,y_train)
y_pred_B = dt_B.predict(X_B_test)

#model performance evaluation used in report
print("Accuracy:", accuracy_score(y_test, y_pred_B))
print(classification_report(y_test, y_pred_B))

plot_cm(y_test, y_pred_B, "Decision Tree B – Confusion Matrix")

fig= plt.figure(figsize=(15,12))
tree.plot_tree(dt_B,
            filled=True,
            feature_names=X_B.columns,
             class_names=['Fail','Pass'] )
plt.title('Decision Tree B')
plt.savefig("tree_B.png")
plt.show()

#Tree C
dt_C = tree.DecisionTreeClassifier(random_state=42, max_depth=15)
dt_C.fit(X_C_train,y_train)
y_pred_C = dt_C.predict(X_C_test)

#model performance evaluation used in report
print("Accuracy:", accuracy_score(y_test, y_pred_C))
print(classification_report(y_test, y_pred_C))

plot_cm(y_test, y_pred_C, "Decision Tree C– Confusion Matrix")

fig= plt.figure(figsize=(15,12))
tree.plot_tree(dt_C,
            filled=True,
            feature_names=X_C.columns,
             class_names=['Fail','Pass'] )
plt.title('Decision Tree C')
plt.savefig("tree_C.png")
plt.show()

# Decision tree prediction compared to actual
pred_pass_rate_A = np.mean(y_pred_A)*100
print(f"Predicted pass rate for Tree A: {pred_pass_rate_A:.3f}%")

pred_pass_rate_B = np.mean(y_pred_B)*100
print(f"Predicted pass rate for Tree B: {pred_pass_rate_B:.3f}%")

pred_pass_rate_C = np.mean(y_pred_C)*100
print(f"Predicted pass rate for Tree C: {pred_pass_rate_C:.3f}%")

actual_pass_rate = np.mean(y_test)*100
print(f"Actual pass rate: {actual_pass_rate:.3f}%")

#/// Task 3 ///
#task1 standardisation only for MLP models using same features as decision trees
scaler_A=preprocessing.MinMaxScaler()

X_A_train_scaled=scaler_A.fit_transform(X_A_train)
X_A_test_scaled=scaler_A.transform(X_A_test)

scaler_B=preprocessing.MinMaxScaler()
X_B_train_scaled=scaler_B.fit_transform(X_B_train)
X_B_test_scaled=scaler_B.transform(X_B_test)

scaler_C=preprocessing.MinMaxScaler()
X_C_train_scaled=scaler_C.fit_transform(X_C_train)
X_C_test_scaled=scaler_C.transform(X_C_test)

#model A
model_A= keras.models.Sequential([ 
keras.layers.Dense(50, activation="relu",input_shape=X_A_train_scaled.shape[1:]),
keras.layers.Dense(20, activation="relu"),
keras.layers.Dense(1,activation='sigmoid')
])

#adam optimizer used over SGD as it offers faster convergence
model_A.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

execution_history_A = model_A.fit(
    X_A_train_scaled,y_train,
    epochs=50, 
validation_data=(X_A_test_scaled,y_test)
)
losses = pd.DataFrame(execution_history_A.history)
losses.plot()
plt.grid(True)
plt.show()

#prediction which are changed to binary 
trgOutputs_A = model_A.predict(X_A_test_scaled)
trgOutputs_binary_A = (trgOutputs_A > 0.5).astype(int)

#model performance evaluation used in report
print("Accuracy:", accuracy_score(y_test, trgOutputs_binary_A))
print(classification_report(y_test, trgOutputs_binary_A))

plot_cm(y_test, trgOutputs_binary_A, "Model A – Confusion Matrix")

plt.figure(figsize = (15,8))
plt.title("MLP Predicted VS actuall pass rate (Model A)")
plt.plot(trgOutputs_binary_A,'r-', label='Predicted')
plt.plot(y_test.values, 'g--', label='Actual')
plt.xlabel("Sample index")
plt.ylabel("Predicted pass (0=Fail, Pass=1)")
plt.legend()
plt.savefig("Model_A.png")
plt.show()

#model B
model_B= keras.models.Sequential([ 
keras.layers.Dense(50, activation="relu",input_shape=X_B_train_scaled.shape[1:]),
keras.layers.Dense(20, activation="relu"),
keras.layers.Dense(1,activation='sigmoid')
])
model_B.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

execution_history_B = model_B.fit(
    X_B_train_scaled,y_train,
    epochs=50, 
validation_data=(X_B_test_scaled,y_test)
)
losses = pd.DataFrame(execution_history_B.history)
losses.plot()
plt.grid(True)
plt.show()

#prediction which are changed to binary 
trgOutputs_B = model_B.predict(X_B_test_scaled)
trgOutputs_binary_B = (trgOutputs_B > 0.5).astype(int)

#model performance evaluation used in report
print("Accuracy:", accuracy_score(y_test, trgOutputs_binary_B ))
print(classification_report(y_test, trgOutputs_binary_B ))

plot_cm(y_test, trgOutputs_binary_B , "Model B– Confusion Matrix")

plt.figure(figsize = (15,8))
plt.title("MLP Predicted VS actuall pass rate (Model B)")
plt.plot(trgOutputs_binary_B,'r-', label='Predicted')
plt.plot(y_test.values, 'g--', label='Actual')
plt.xlabel("Sample index")
plt.ylabel("Predicted pass (0=Fail, Pass=1)")
plt.legend()
plt.savefig("Model_B.png")
plt.show()

#model C
model_C= keras.models.Sequential([ 
keras.layers.Dense(50, activation="relu",input_shape=X_C_train_scaled.shape[1:]),
keras.layers.Dense(20, activation="relu"),
keras.layers.Dense(1,activation='sigmoid')
])
model_C.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

execution_history_C = model_C.fit(
    X_C_train_scaled,y_train,
    epochs=50, 
validation_data=(X_C_test_scaled,y_test)
)
losses = pd.DataFrame(execution_history_C.history)
losses.plot()
plt.grid(True)
plt.show()

#prediction which are changed to binary 
trgOutputs_C = model_C.predict(X_C_test_scaled)
trgOutputs_binary_C = (trgOutputs_C > 0.5).astype(int)

#model performance evaluation used in report
print("Accuracy:", accuracy_score(y_test, trgOutputs_binary_C))
print(classification_report(y_test, trgOutputs_binary_C))

plot_cm(y_test, trgOutputs_binary_C , "Model C– Confusion Matrix")

plt.figure(figsize = (15,8))
plt.title("MLP Predicted VS actuall pass rate (Model C)")
plt.plot(trgOutputs_binary_C,'r-', label='Predicted')
plt.plot(y_test.values, 'g--', label='Actual')
plt.xlabel("Sample index")
plt.ylabel("Predicted pass (0=Fail, Pass=1)")
plt.legend()
plt.savefig("Model_C.png")
plt.show()

#MLP prediction comapared to actual
pred_pass_rate_model_A = np.mean(trgOutputs_binary_A)*100
print(f"Predicted pass rate for MLP Model A: {pred_pass_rate_model_A:.3f}%")

pred_pass_rate_model_B = np.mean(trgOutputs_binary_B)*100
print(f"Predicted pass rate for MLP Model B: {pred_pass_rate_model_B:.3f}%")

pred_pass_rate_model_C = np.mean(trgOutputs_binary_C)*100
print(f"Predicted pass rate for MLP Model C: {pred_pass_rate_model_C:.3f}%")

actual_pass_rate = np.mean(y_test)*100
print(f"Actual pass rate: {actual_pass_rate:.3f}%")