import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load data from Excel file
file_path = 'ProjectInfo.xlsx'
df = pd.read_excel(r'C:\Users\cthor\OneDrive\Documents\School\Fall 2023\DataMining\ProjectInfo.xlsx')

# Assuming column 'A' is the target variable
y = df['Price']  # Target variable from column A

# Select features
features = ['Beds', 'baths', 'SqFT', 'Zip', 'Garage']

# Ensure all selected features are present in the DataFrame
for feature in features:
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in the DataFrame.")

X = df[features]  # Features from columns B to F

# One-hot encoding for categorical variables
X = pd.get_dummies(X, columns=['Zip', 'Garage'], drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values with the mean
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Print columns before one-hot encoding in the training set
print("Columns before one-hot encoding in the training set:", X_train.columns)

# Drop columns with NaN values before one-hot encoding
X_train = X_train.dropna(axis=1)
X_test = X_test.dropna(axis=1)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Print the coefficients for linear regression
coefficients_linear = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': linear_model.coef_})
print("Linear Regression Coefficients:")
print(coefficients_linear)

# Make predictions on the test set with linear regression
predictions_linear = linear_model.predict(X_test_scaled)

# Evaluate the linear regression model
mse_linear = mean_squared_error(y_test, predictions_linear)
print(f'Mean Squared Error on Test Set (Linear Regression): {mse_linear}')

# Initialize and train the Lasso model with regularization
alpha = 0.1  # Adjust the regularization strength
lasso_model = Lasso(alpha=alpha)
lasso_model.fit(X_train_scaled, y_train)

# Print the coefficients for Lasso
coefficients_lasso = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': lasso_model.coef_})
print("\nLasso Coefficients:")
print(coefficients_lasso)

# Make predictions on the test set with Lasso
predictions_lasso = lasso_model.predict(X_test_scaled)

# Evaluate the Lasso model
mse_lasso = mean_squared_error(y_test, predictions_lasso)
print(f'Mean Squared Error on Test Set (Lasso): {mse_lasso}')

# Input your own test values
custom_test_values = {
    'Beds': 5,
    'baths': 5,
    'SqFT': 4672,
    'Zip': '36605',  # Replace with an actual zipcode
    'Garage': 0  # Replace with 0 or 1 depending on garage presence
}

# Create a DataFrame with the custom test values
custom_test_df = pd.DataFrame([custom_test_values])

# Ensure all selected features are present in the custom test DataFrame
for feature in features:
    if feature not in custom_test_df.columns:
        raise ValueError(f"Feature '{feature}' not found in the custom test DataFrame.")

# One-hot encoding for categorical variables in the custom test set
custom_test_df = pd.get_dummies(custom_test_df, columns=['Zip', 'Garage'], drop_first=True)

# Drop columns with NaN values before one-hot encoding
custom_test_df = custom_test_df.dropna(axis=1)

# Ensure the custom test DataFrame has the same columns as the training data after one-hot encoding
custom_test_df = custom_test_df.reindex(columns=X_train.columns, fill_value=0)

# Impute missing values with the mean
custom_test_df = custom_test_df.fillna(X_train.mean())

# Ensure the custom test DataFrame has the same columns as the training data after one-hot encoding
if set(custom_test_df.columns) != set(X_train.columns):
    raise ValueError("Columns after one-hot encoding in the custom test set do not match the training set.")

# Feature scaling for the custom test set
custom_test_scaled = scaler.transform(custom_test_df)

# Make predictions with both linear regression and Lasso models
custom_prediction_linear = linear_model.predict(custom_test_scaled)
custom_prediction_lasso = lasso_model.predict(custom_test_scaled)

# Round the predicted values to the nearest cent
rounded_prediction_linear = round(custom_prediction_linear[0], 2)
rounded_prediction_lasso = round(custom_prediction_lasso[0], 2)

# Manually adjust Lasso prediction based on coefficients for the given zip code and garage presence
zip_feature_linear = f'Zip_{custom_test_values["Zip"]}'
garage_feature_linear = f'Garage_{custom_test_values["Garage"]}'

zip_coefficient_linear = 0
garage_coefficient_linear = 0

if zip_feature_linear in coefficients_linear['Feature'].values:
    zip_coefficient_linear = coefficients_linear.loc[coefficients_linear['Feature'] == zip_feature_linear, 'Coefficient'].values[0]

if garage_feature_linear in coefficients_linear['Feature'].values:
    garage_coefficient_linear = coefficients_linear.loc[coefficients_linear['Feature'] == garage_feature_linear, 'Coefficient'].values[0]

adjusted_prediction_linear = rounded_prediction_linear + zip_coefficient_linear + garage_coefficient_linear

# Manually adjust Lasso prediction based on coefficients for the given zip code and garage presence
zip_feature_lasso = f'Zip_{custom_test_values["Zip"]}'
garage_feature_lasso = f'Garage_{custom_test_values["Garage"]}'

zip_coefficient_lasso = 0
garage_coefficient_lasso = 0

if zip_feature_lasso in coefficients_lasso['Feature'].values:
    zip_coefficient_lasso = coefficients_lasso.loc[coefficients_lasso['Feature'] == zip_feature_lasso, 'Coefficient'].values[0]

if garage_feature_lasso in coefficients_lasso['Feature'].values:
    garage_coefficient_lasso = coefficients_lasso.loc[coefficients_lasso['Feature'] == garage_feature_lasso, 'Coefficient'].values[0]

adjusted_prediction_lasso = rounded_prediction_lasso + zip_coefficient_lasso + garage_coefficient_lasso

# Print the rounded and adjusted predicted values
print(f'Predicted house price for custom test values before Coefficient Adjustments: ${rounded_prediction_linear}')
print(f'Adjusted Predicted house price for custom test values (Linear Regression): ${round(adjusted_prediction_linear, 2)}')
print(f'Adjusted Predicted house price for custom test values (Lasso): ${round(adjusted_prediction_lasso, 2)}')