import numpy as np
import matplotlib.pyplot as plt

# Training data
X_train = np.array([1, 2, 3, 4, 5, 10, 11, 12, 13])
Y_train = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])

# Calculate means
X_mean = np.mean(X_train)
Y_mean = np.mean(Y_train)

# Calculate the slope (b1) and intercept (b0)
numerator = np.sum((X_train - X_mean) * (Y_train - Y_mean))
denominator = np.sum((X_train - X_mean) ** 2)
b1 = numerator / denominator
b0 = Y_mean - b1 * X_mean

# Regression function
def predict(X):
    return b0 + b1 * X

# Test points
X_test = np.array([2.4, 5.5, 3.9])
Y_test = np.array([0, 1, 0])

# Output the regression equation
print(f"Regression equation: Y = {b1:.2f}X + {b0:.2f}")

# Calculate error rates
# Predict the values for the training set
Y_train_pred = predict(X_train)
Y_train_pred_class = (Y_train_pred >= 0.5).astype(int)

# Calculate the error rate for the training set
train_error_rate = np.mean(Y_train_pred_class != Y_train)
print(f"Training set error rate: {train_error_rate:.2f}")

# Predict the values for the test set
Y_test_pred = predict(X_test)
Y_test_pred_class = (Y_test_pred >= 0.5).astype(int)

# Calculate the error rate for the test set
test_error_rate = np.mean(Y_test_pred_class != Y_test)
print(f"Test set error rate: {test_error_rate:.2f}")

# Plotting
plt.scatter(X_train, Y_train, color='blue', label='Training data')
plt.plot(X_train, predict(X_train), color='red', label='Regression line')
plt.scatter(X_test, Y_test, color='green', label='Test data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()