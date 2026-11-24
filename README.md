# Credit Risk Classification | Machine Learning & Streamlit

## 📌 Project Overview

This project aims to develop an effective Machine Learning classification model to assess the risk of granting credit to a customer and predict whether a loan application should be approved or rejected.

The project follows a complete Machine Learning workflow, from data collection and exploration to model development, evaluation, and saving the best-performing model. An interactive web application was also developed using Streamlit to demonstrate the model's predictions.

## 🎯 Project Objectives

- Retrieve and verify the dataset.
- Explore and understand the data through Exploratory Data Analysis (EDA).
- Clean and preprocess the data.
- Perform data transformation and feature engineering.
- Develop and compare different Machine Learning classification models.
- Evaluate model performance and select the best-performing model.
- Save the selected model for future predictions.
- Provide interpretations and justifications for the main analytical and modeling decisions.

## 📊 Dataset

The project uses `Dataset.csv`, which contains information about loan applicants, including:

- Personal information: Gender, Marital Status, Dependents and Education.
- Financial information: Applicant Income, Co-applicant Income and Loan Amount.
- Credit information: Credit History and Loan Term.
- Property information: Property Area.

**Target Variable:** `Loan_Status` (Y: Approved / N: Rejected).

## 🛠️ Tools & Technologies

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Jupyter Notebook
- Streamlit

## 📂 Project Structure

| File | Description |
|---|---|
| `Dataset.csv` | Original dataset |
| `Projet-machine-learning.ipynb` | Complete data analysis and Machine Learning workflow |
| `model_RL.pkl` | Saved trained model |
| `app.py` | Interactive Streamlit application |

## 🚀 Run the Application

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
```

Launch the application:

```bash
streamlit run app.py
```

## 👩‍💻 Author

**Hoda Boustany**  
Master 2 – Data Science & Actuariat

**Project Category:** Machine Learning | Credit Risk Analysis
