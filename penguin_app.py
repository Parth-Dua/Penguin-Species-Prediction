# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

@st.cache()
def prediction(model,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex):

	y_pred = model.predict([[island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex]])

	if y_pred ==0:
		return 'Adelie'
	elif y_pred ==1:
		return 'Chinstrap'
	else:
		return 'Gentoo'

#Design 
st.sidebar.title("Penguin Species Prediction Application")

l1 =['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
l2 = []
for i in range(4):

	l2.append(st.sidebar.slider(l1[i] , 0.00 , 10.00))


# ISLAND
island1 = st.sidebar.selectbox('What is island of penguin ?',('Biscoe', 'Dream' ,'Torgersen'))
if island1 == 'Biscoe':
	island=0
elif island1 == 'Dream':
	island=1
else:
	island=2

# SEX 
sex = st.sidebar.selectbox('What is Sex of penguin ?',('Male', 'Female'))
if sex == 'Male':
	sex=0
else:
	sex=1

st.sidebar.subheader("Choose Classifier")
model_chosen = st.sidebar.selectbox("Classifier" , ("LogisticRegression", 'SVC', 'RandomForestClassifier'))
if model_chosen == 'LogisticRegression':
	model = log_reg
elif model_chosen == 'SVC':
	model = svc_model
elif model_chosen == 'RandomForestClassifier':
	model = rf_clf

if st.sidebar.button("Predict"):
	predicted_value = prediction(model ,island ,l2[0],l2[1],l2[2],l2[3],sex)

	st.write("Species predicted ", predicted_value)	
	st.write("Accuracy of this model ", model.score(X_train,y_train))
