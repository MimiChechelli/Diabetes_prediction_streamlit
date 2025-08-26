import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns


diabetes_raw_data = pd.read_csv('diabetes.csv')

st.title('Diabetes Checkup')
st.sidebar.header('Select the Patient Data:')
st.subheader('Training Data Stats')
st.write(diabetes_raw_data.describe())

target = 'Outcome'
x = diabetes_raw_data.drop([target], axis = 1)
y = diabetes_raw_data[target]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 23)
rf  = RandomForestClassifier(random_state=16)
rf.fit(x_train, y_train)
st.write('Accuracy: ' + str(round(accuracy_score(y_test, rf.predict(x_test)) * 100, 2)) + '%')

# Sliders
def user_report():
  pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
  glucose = st.sidebar.slider('Glucose', 0,200, 120 )
  bp = st.sidebar.slider('Blood Pressure', 0,122, 70 )
  skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
  insulin = st.sidebar.slider('Insulin', 0,846, 79 )
  bmi = st.sidebar.slider('BMI', 0,67, 20 )
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
  age = st.sidebar.slider('Age', 21,88, 33 )

  user_report_data = {
      'Pregnancies':pregnancies,
      'Glucose':glucose,
      'BloodPressure':bp,
      'SkinThickness':skinthickness,
      'Insulin':insulin,
      'BMI':bmi,
      'DiabetesPedigreeFunction':dpf,
      'Age':age
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()
st.subheader('Sected Patient Data')
st.write(user_data)

user_result = rf.predict(user_data)

st.title('Visualised Patient Report')

if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'

output=''
if user_result[0]==0:
  output = 'Pacient not Diabetic'
else:
  output = 'Pacient are Diabetic'
st.subheader(output)



st.header('Pregnancy count Graph (Others vs Patient)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = diabetes_raw_data, hue = 'Outcome', palette = 'Greens')
ax2 = sns.scatterplot(x = user_data['Age'], y = user_data['Pregnancies'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)


st.header('Glucose Value Graph (Others vs Patient)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = diabetes_raw_data, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['Age'], y = user_data['Glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)


st.header('Blood Pressure Value Graph (Others vs Patient)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = diabetes_raw_data, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = user_data['Age'], y = user_data['BloodPressure'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)


st.header('Skin Thickness Value Graph (Others vs Patient)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = diabetes_raw_data, hue = 'Outcome', palette='Blues')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['SkinThickness'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)


st.header('Insulin Value Graph (Others vs Patient)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = diabetes_raw_data, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['Age'], y = user_data['Insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)


st.header('BMI Value Graph (Others vs Patient)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = diabetes_raw_data, hue = 'Outcome', palette='rainbow')
ax12 = sns.scatterplot(x = user_data['Age'], y = user_data['BMI'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)


st.header('DPF Value Graph (Others vs Patient)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = diabetes_raw_data, hue = 'Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['DiabetesPedigreeFunction'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)
