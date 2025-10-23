# Loan_Approval_Prediction
A Streamlit web app that predicts whether a loan will be approved using a trained Logistic Regression model. Built with Python, scikit-learn and deployed with Streamlit.

# 1. Project Overview
## This project predicts loan approval using historical loan application data. The workflow includes:

* Loading and cleaning the dataset
* Exploratory data analysis (EDA)
* Feature engineering and transformations
* Training multiple classification models
* Selecting the best model (Logistic Regression) and saving it as loan_approval_model.pkl
* Building a Streamlit app (app2.py) that loads the saved model and returns predictions based on user input

The repository contains a Jupyter notebook with the full modeling pipeline and a Streamlit app to interact with the saved model.

# 2. Repository Structure
```
Loan_Approval_Prediction/
├─ Dataset.csv                           # Raw dataset used for training
├─ Loan_Approval_Pred_Proj_ML.ipynb      # Jupyter notebook with full pipeline
├─ app2.py                               # Streamlit app file
├─ loan_approval_model.pkl               # Saved (pickled) best model
├─ Output_Streamlit_1.jpg                # Screenshot(s) of the Streamlit UI
├─ Output_Streamlit_2.jpg
├─ Output_Streamlit_3.jpg
└─ README.md                             # (this file)
```
# 3. Tech Stack
* Python 3.8+ (recommended)
* pandas
* numpy
* scikit-learn
* matplotlib / seaborn (for EDA/plots)
* pickle (for model serialization)
* Streamlit (for web UI)

# 4. Dataset
* **File**: Dataset.csv
* **Description**: Contains historical loan application records and their approval status (Loan_Status). Typical columns include Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, and Loan_Status.

# 5. Quickstart — Run the App Locally
## 1. Clone the repository
```
git clone https://github.com/Krishna35-it/Loan_Approval_Prediction.git
cd Loan_Approval_Prediction
```

## 2. Create and activate a virtual environment (recommended)
### Windows (cmd):
```
python -m venv venv
venv\Scripts\activate
```
### macOS / Linux:
```
python3 -m venv venv
source venv/bin/activate
```

## 3.Install dependencies
install packages manually: 
```
pip install streamlit scikit-learn pandas numpy pickle
```

## 4.Run the Streamlit app
```
streamlit run app2.py
```

# 6. Reproduce the Model Training (Notebook)
The notebook `Loan_Approval_Pred_Proj_ML.ipynb` contains the full reproducible pipeline. Key steps to run the notebook:

1) Open the notebook in Jupyter Lab / Notebook or VS Code.
2) Make sure `Dataset.csv` is in the same working directory.
3) Follow and run cells in order — sections include:
     * Import libraries
     * Load dataset
     * Exploratory Data Analysis (plots, missing values)
     * Data preprocessing (handling missing values, encoding categorical variables)
     * Feature scaling / transformations
     * Model training and evaluation
     * Saving the best model to `loan_approval_model.pkl`
### Tip: If runtime fails due to package versions, create a clean environment and reinstall required package versions.

# 7. Data Preprocessing & Feature Engineering
* Inspect the dataset.
* Handle missing values.
* Feature engineering.
* Encoding categorical variables.
* Train/test split.  

# 8. Models Trained & Selection
* Logistic Regression (chosen best-model)
* Decision Tree Classifier
* Random Forest Classifier
* K-Nearest Neighbors

# 9. Evaluation Metrics(Accuracy) & Results
* `Logistic Regression: 78.86`  
* `Decision Tree: 65.85`  
* `Random Forest: 76.42`
* `KNN: 77.23`

# 10. Deployment
     * Using Streamlit Application
     * Allows User to Enter the values
     * Predicts whether loan is approved or not

