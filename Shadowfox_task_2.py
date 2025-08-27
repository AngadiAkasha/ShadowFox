#importing the required package based on the problem numpy ,pandas,sklearn
# we use the some library like pandas ,numpy,sklearn
import pandas as g
import numpy as n
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#Load data of the information of the problem give the document
d = "HousingData (2).csv"
r = g.read_csv(d)
print("starting value of the data :",r.shape)
print("Missing values per columns :\n",r.isnull().sum())
num_features = r.select_dtypes(include=[n.number])
imputer = IterativeImputer(random_state = 42)
im_v = imputer.fit_transform(num_features)
r_imputed = g.DataFrame(im_v,columns=num_features.columns)
print("\n Missing values after imputation\n",r_imputed.isnull().sum().sum())
#3.Features /Target split
X = r_imputed.drop("MEDV",axis=1)
y = r_imputed["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 5. Standardization
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 6. Define models
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
}
results = {}
fitted_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    results[name] = {
        "R² Score": r2_score(y_test, preds),
        "RMSE": n.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
    }
    fitted_models[name] = model

# Convert to DataFrame for readability
results_df = g.DataFrame(results).T
print("\nModel Performance:\n")
print(results_df)
# 8. Feature Importance (from best tree-based model)
# -----------------------------
best_model = fitted_models["Gradient Boosting"]
importances = best_model.feature_importances_
sorted_idx = n.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[sorted_idx], align="center")
plt.xticks(range(len(importances)), X.columns[sorted_idx], rotation=45)
plt.title("Boston Housing Prediction - Feature Importances")
plt.tight_layout()
plt.show()
'''
SOME USEFUL information abbrevations 
1.RM = Average number of rooms per Dwelling.
2.LSTAT =  Lower status of the Population.
3.DIS = Weighted distances to five Boston employment centers.
4.PTRATIO = Pupil–teacher ratio by town.
5.NOX = Nitric oxides concentration.
6.CRIM = Per capita crime rate by town.
7.TAX =Full-value property-tax rate per $10,000.
8.ZN = Proportion of residential land zoned for lots over 25,000 sq.ft.
9.INDUS = Proportion of non-retail business acres per town.
10.CHAS = Charles River dummy variable (1 if tract bounds river; 0 otherwise).
'''