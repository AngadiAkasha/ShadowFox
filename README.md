# 🏠 Boston Housing Price Prediction
### Shadow Fox Internship – Machine Learning Project

---

## 📌 Problem Statement
This project aims to predict **Boston house prices** using machine learning regression models.  
The dataset includes housing-related features such as crime rate, average number of rooms, pupil–teacher ratio, and more.  
By applying preprocessing, missing value imputation, feature scaling, and multiple regression techniques, the project evaluates model performance and identifies the **most important factors influencing house prices**.

---

## 📂 Project Structure
├── HousingData (2).csv # Dataset containing housing features and target values (MEDV)
├── boston_housing.py # Python script for preprocessing, training, evaluation, and visualization
├── README.md # Project documentation

---

## 📊 Dataset Information
- **Target Variable:**  
  - `MEDV` – Median value of owner-occupied homes

- **Features Include:**  
  - Crime rate (CRIM)  
  - Average number of rooms (RM)  
  - Pupil–teacher ratio (PTRATIO)  
  - Property tax rate (TAX)  
  - Accessibility to highways (RAD)  
  - Other socio-economic indicators  

---

## ⚙️ Technologies Used
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  

---

## 📥 Requirements / Installation
Install the required libraries using the following command:

```bash
pip install numpy pandas scikit-learn matplotlib
flowchart TD
    A[Load Dataset] --> B[Check Missing Values]
    B --> C[Impute Missing Values using IterativeImputer]
    C --> D[Split Features & Target]
    D --> E[Train-Test Split (80/20)]
    E --> F[Standardize Features using StandardScaler]
    F --> G[Train Regression Models]
    G --> H[Evaluate Models]
    H --> I[Select Best Model]
    I --> J[Feature Importance Analysis]
    J --> K[Visualization & Insights]
🧠 Machine Learning Models Used

Linear Regression

Random Forest Regressor

Gradient Boosting Regressor

📈 Model Evaluation Metrics

The models are evaluated using:

R² Score

Root Mean Squared Error (RMSE)

Mean Absolute Error (MAE)

The best-performing model is selected based on these metrics.

🔍 Feature Importance

Tree-based models are used to determine feature importance, helping identify which housing attributes most strongly affect property prices.
The results are visualized using bar plots for clear interpretation.

✅ Results & Key Insights

Successfully predicted Boston house prices using multiple regression models

Compared model performance using standard evaluation metrics

Identified key features influencing housing prices

Gained hands-on experience in data preprocessing and ML workflows

🚀 Future Enhancements

Hyperparameter tuning for improved accuracy

Cross-validation for robust performance evaluation

Model deployment using Flask or Streamlit

Implementation of advanced models like XGBoost
