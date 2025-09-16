import streamlit as st
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Data", "diabetes.csv")

diabetes_raw_data = pd.read_csv(csv_path)

cols = ["BloodPressure", "Glucose", "Insulin", "BMI", "SkinThickness"]
diabetes_raw_data = diabetes_raw_data[~(diabetes_raw_data[cols] == 0).any(axis=1)]


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
importances = rf.feature_importances_

feat_importances = pd.DataFrame({
    "Feature": x_train.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.subheader('Feature Importance')
st.bar_chart(feat_importances.set_index('Feature'))
st.write('Accuracy: ' + str(round(accuracy_score(y_test, rf.predict(x_test)) * 100, 2)) + '%')

# Sliders
def user_report():
  pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3 )
  glucose = st.sidebar.slider('Glucose', 50, 200, 120 )
  bp = st.sidebar.slider('Blood Pressure', 24, 110, 70 )
  skinthickness = st.sidebar.slider('Skin Thickness', 7, 63, 20 )
  insulin = st.sidebar.slider('Insulin', 14, 846, 79 )
  bmi = st.sidebar.slider('BMI', 18, 67, 20 )
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.45, 0.47 )
  age = st.sidebar.slider('Age', 21, 81, 33 )

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
  color = 'green'
else:
  color = 'red'

output=''
if user_result[0]==0:
  output = 'Pacient not Diabetic'
else:
  output = 'Pacient are Diabetic'
st.subheader(output)

diabetes_raw_data["Outcome"] = diabetes_raw_data["Outcome"].replace({1: "Diabetic", 0: "Not diabetic"})
palette = {
    "Diabetic": "#344A53",
    "Not diabetic": "#B0C4DE"
}


st.header('Pregnancy Distribuition Graph (Others vs Patient)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(
    x = 'Glucose', 
    y = 'Pregnancies', 
    data = diabetes_raw_data, 
    hue = 'Outcome', 
    palette = palette)
ax2 = sns.scatterplot(
    x = user_data['Glucose'], 
    y = user_data['Pregnancies'], 
    s = 100, 
    color = color)
sns.regplot(
    x='Glucose',
    y='Pregnancies',
    data=diabetes_raw_data,
    scatter=False,        # não desenha os pontos de novo
    ax=ax1,
    color="#0E2035"         # cor da linha de tendência
)
plt.xticks(np.arange(50,211,15))
plt.yticks(np.arange(0,19,2))
plt.title('')
st.pyplot(fig_preg)


st.header('Age Distribuition Graph (Others vs Patient)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(
    x = 'Glucose', 
    y = 'Age', 
    data = diabetes_raw_data, 
    hue = 'Outcome' , 
    palette=palette)
ax4 = sns.scatterplot(
    x = user_data['Glucose'], 
    y = user_data['Age'], 
    s = 100, 
    color = color)
sns.regplot(
    x='Glucose',
    y='Age',
    data=diabetes_raw_data,
    scatter=False,
    ax=ax3,
    color="#0E2035"
)
plt.xticks(np.arange(50,211,15))
plt.yticks(np.arange(20,86,5))
plt.ylim(19,85)
plt.title('')
st.pyplot(fig_glucose)


st.header('Blood Pressure Distribuition Graph (Others vs Patient)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(
    x = 'Glucose', 
    y = 'BloodPressure', 
    data = diabetes_raw_data, 
    hue = 'Outcome', 
    palette=palette)
ax6 = sns.scatterplot( 
    x = user_data['Glucose'], 
    y = user_data['BloodPressure'], 
    s = 100, 
    color = color)
sns.regplot(
    x='Glucose',
    y='BloodPressure',
    data=diabetes_raw_data,
    scatter=False,
    ax=ax5,
    color="#0E2035"
)
plt.xticks(np.arange(50,211,15))
plt.yticks(np.arange(20,130,10))
plt.title('')
st.pyplot(fig_bp)


st.header('Skin Thickness Distribuition Graph (Others vs Patient)')
fig_st = plt.figure()
ax7 = sns.scatterplot(
    x = 'Glucose', 
    y = 'SkinThickness', 
    data = diabetes_raw_data, 
    hue = 'Outcome', 
    palette=palette)
ax8 = sns.scatterplot(
    x = user_data['Glucose'], 
    y = user_data['SkinThickness'], 
    s = 100, 
    color = color)
sns.regplot(
    x='Glucose',
    y='SkinThickness',
    data=diabetes_raw_data,
    scatter=False,
    ax=ax7,
    color="#0E2035"
)
plt.xticks(np.arange(50,211,15))
plt.yticks(np.arange(0,65,5))
plt.title('')
st.pyplot(fig_st)


st.header('Insulin Value Graph (Others vs Patient)')
fig_i = plt.figure()
ax9 = sns.scatterplot(
    x = 'Glucose', 
    y = 'Insulin', 
    data = diabetes_raw_data, 
    hue = 'Outcome', 
    palette=palette)
ax10 = sns.scatterplot(
    x = user_data['Glucose'], 
    y = user_data['Insulin'], 
    s = 100, 
    color = color)
sns.regplot(
    x='Glucose',
    y='Insulin',
    data=diabetes_raw_data,
    scatter=False,
    ax=ax9,
    color="#0E2035"
)
plt.xticks(np.arange(50,211,15))
plt.yticks(np.arange(0,851,50))
plt.ylim(0,850)
plt.title('')
st.pyplot(fig_i)


st.header('BMI Value Graph (Others vs Patient)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(
    x = 'Glucose', 
    y = 'BMI', 
    data = diabetes_raw_data, 
    hue = 'Outcome', 
    palette=palette)
ax12 = sns.scatterplot(
    x = user_data['Glucose'], 
    y = user_data['BMI'], 
    s = 100, 
    color = color)
sns.regplot(
    x='Glucose',
    y='BMI',
    data=diabetes_raw_data,
    scatter=False,
    ax=ax11,
    color="#0E2035"
)
plt.xticks(np.arange(50,211,15))
plt.yticks(np.arange(15,71,5))
plt.title('')
st.pyplot(fig_bmi)


st.header('DPF Value Graph (Others vs Patient)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(
    x = 'Glucose', 
    y = 'DiabetesPedigreeFunction', 
    data = diabetes_raw_data, 
    hue = 'Outcome', 
    palette=palette)
ax14 = sns.scatterplot(
    x = user_data['Glucose'], 
    y = user_data['DiabetesPedigreeFunction'], 
    s = 100, 
    color = color)
sns.regplot(
    x='Glucose',
    y='DiabetesPedigreeFunction',
    data=diabetes_raw_data,
    scatter=False,
    ax=ax13,
    color="#0E2035"
)
plt.xticks(np.arange(50,211,15))
plt.yticks(np.arange(0,3,0.2))
plt.title('')
st.pyplot(fig_dpf)
