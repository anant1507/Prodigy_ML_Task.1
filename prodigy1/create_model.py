import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Generate some sample data for demonstration
np.random.seed(0)
n_samples = 100
square_feet = np.random.normal(2000, 500, n_samples)
bedrooms = np.random.randint(2, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
price = 100000 + 300 * square_feet + 20000 * bedrooms + 15000 * bathrooms + np.random.normal(0, 10000, n_samples)

# Create a feature matrix X
X = np.column_stack((square_feet, bedrooms, bathrooms))

# Create a target vector y
y = price

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Save the model to a file named 'model.pkl'
joblib.dump(model, 'model.pkl')

print("Model saved as model.pkl")
