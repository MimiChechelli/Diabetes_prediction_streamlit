# Diabetes Prediction Web App

## About the Data

This dataset is used for predicting diabetes based on medical measurements. The features include:

* **Pregnancies**: Number of pregnancies
* **Glucose**: Glucose level
* **BloodPressure**: Blood pressure measurement
* **SkinThickness**: Skinfold thickness
* **Insulin**: Insulin level
* **BMI**: Body Mass Index
* **DiabetesPedigreeFunction**: Diabetes pedigree function
* **Age**: Age of the patient
* **Outcome**: Target variable (0 = No diabetes, 1 = Diabetes)

## Requirements

* Python (any version up to 3.11)
* Libraries: `streamlit`, `pandas`, `scikit-learn`

You can install the required libraries using:

```bash
pip install streamlit pandas scikit-learn
```

## Usage

Run the app using the command:

```bash
streamlit run app.py
```

This will open the interactive web app in your default browser. You can input patient data and get diabetes predictions in real time.

# Repository 
```bash
📂 Diabetes predicion streamlit
├── 📂 data/ 
│   └── 📄 data.csv
│
├── 📂 Notebooks/ 
│   └── test.ipynb
│
├── 📂 scr/
│   └── .py
│
├── 📂 utils/        
│   └── .py
│
├── 📄 app.py
│
└── 📄 README.md      # Documentação do projeto

```

# References:
- [Deep learning approach for diabetes prediction using PIMA Indian dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC7270283/)