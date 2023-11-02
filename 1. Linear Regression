import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generating random data for demonstration purposes
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit and transform the scaler on the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Create a linear regression model
model = linear_model.LinearRegression()

# Fit the model on the scaled training data
model.fit(X_train_scaled, y_train)

# Make predictions on the scaled testing data
y_predicted = model.predict(X_test_scaled)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_predicted)
print("Mean squared error:", mse)

# Print the model's coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Plot the original target values and the regression line
plt.scatter(X_test_scaled, y_test, color='black', label='True Values')
plt.plot(X_test_scaled, y_predicted, color='blue', linewidth=3, label='Predicted Values')
plt.xlabel("Feature (Scaled)")
plt.ylabel("Target")
plt.title("True Values vs. Predicted Values")
plt.legend()
plt.show()
