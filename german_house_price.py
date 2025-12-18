import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


data = pd.read_csv("data/raw/german_housing_clean.csv")
target = 'obj_purchasePrice'
data = data.dropna(subset=[target])


categorical_features = ['obj_regio1', 'obj_heatingType', 'obj_condition', 'obj_immotype', 'obj_barrierFree']
numeric_features = ['obj_livingSpace', 'obj_noRooms', 'obj_yearConstructed']


data[categorical_features] = data[categorical_features].fillna('missing')
data[numeric_features] = data[numeric_features].fillna(data[numeric_features].median())


current_year = 2025
data['building_age'] = current_year - data['obj_yearConstructed']
data['price_per_sqm'] = data[target] / data['obj_livingSpace']


lower = data['price_per_sqm'].quantile(0.01)
upper = data['price_per_sqm'].quantile(0.99)
data = data[(data['price_per_sqm'] >= lower) & (data['price_per_sqm'] <= upper)]


features = categorical_features + numeric_features + ['building_age']
X = data[features]
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features + ['building_age'])
    ]
)


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])


param_dist = {
    'regressor__n_estimators': [300, 500, 700],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [3, 4, 5, 6],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__subsample': [0.7, 0.8, 1.0]
}

search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist,
    n_iter=20, cv=3, scoring='r2', n_jobs=-1, random_state=42, verbose=1
)

search.fit(X_train, y_train)


model_optimized = search.best_estimator_
y_train_pred = model_optimized.predict(X_train)
y_test_pred = model_optimized.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("Best Parameters:", search.best_params_)
print("Train R²:", r2_score(y_train, y_train_pred))
print("Test R²:", r2_score(y_test, y_test_pred))
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

joblib.dump(model_optimized, 'models/german_price_model_optimized.pkl')
print("Optimized pipeline saved as 'models/german_price_model_optimized.pkl'")


sample = pd.DataFrame([{
    'obj_regio1': 'Berlin',
    'obj_heatingType': 'central_heating',
    'obj_condition': 'good',
    'obj_immotype': 'apartment',
    'obj_barrierFree': 'yes',
    'obj_livingSpace': 75,
    'obj_noRooms': 3,
    'obj_yearConstructed': 1995,
    'building_age': current_year - 1995
}])
predicted_price = model_optimized.predict(sample)
print("Predicted Price for sample:", predicted_price[0])


os.makedirs('github_results', exist_ok=True)


plt.figure(figsize=(8,5))
sns.histplot(data['obj_purchasePrice'], bins=50, kde=True)
plt.title("Distribution of Purchase Price")
plt.xlabel("Price (€)")
plt.ylabel("Count")
plt.savefig('output/target_distribution.png')
plt.close()

plt.figure(figsize=(8,5))
sns.histplot(data['price_per_sqm'], bins=50, kde=True, color='green')
plt.title("Price per Square Meter")
plt.xlabel("€/m²")
plt.ylabel("Count")
plt.savefig('output/price_per_sqm.png')
plt.close()


plt.figure(figsize=(8,6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.xlabel("Actual Price (€)")
plt.ylabel("Predicted Price (€)")
plt.title("Actual vs Predicted Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.savefig('output/actual_vs_predicted.png')
plt.close()

# Feature Importance
regressor = model_optimized.named_steps['regressor']
cat_names = model_optimized.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
num_names = numeric_features + ['building_age']
all_features = np.concatenate([cat_names, num_names])
importances = regressor.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)


top_features = feature_importance_df.head(10)
print("\nTop 10 Features by Importance:\n", top_features)


plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.savefig('output/top_10_feature_importances.png')
plt.close()
