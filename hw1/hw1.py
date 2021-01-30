from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# I pledge my honor that I have abided by the Steven's honor system. - Stephen Szemis

# Constants
test_size = 0.5
n = 8000 # number of iterations
training_rate = 0.001

# Load data
iris = load_iris()

# Gradient of loss (binary cross-entropy) is 
# vectorized is quiet simple. 
# This was very confusing to derive...
def derivative(X, Y, theta):
    # Our current prediction of class labels
    # Based on P(Y=1) = 1 / 1 + e^{-X * theta}
    pred = 1 / (1 + np.exp(-np.dot(X, theta)))
    temp = np.subtract(pred.T, Y)
    return np.dot(X.T, temp.T)

# Takes a default set of weights and uses gradient decent to find
# optimal configuration.
# Params:
# Y -> training targets
# X -> training data
# max_iter -> max iterations
# rate -> training rate
# (option) theta -> initial weights, defaults to [1] * dimesions
def train(X, Y, max_iter, rate, theta):
    curr_iter = 0
    while (curr_iter < max_iter):
        z = (rate * derivative(X, Y, theta))
        theta = theta - z
        curr_iter += 1
    return theta

# Ignore all but sepal length and width
features = iris['data'][ : , : 2]

# 1 for virginica, 0 otherwise
labels = iris['target']
labels[labels < 2] = 0
labels[labels == 2] = 1

# for n in range(1000, 11000, 1000):

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=test_size)
initial_theta = np.array([[1], [1]])
model = train(np.array(X_train), np.array([Y_train]), n, training_rate, initial_theta)

# Sigmoid
result = 1 / (1 + np.exp(-np.dot(X_test, model)))

result[result >= 0.5] = 1
result[result <= 0.5] = 0
accuracy = 1 - (np.count_nonzero(result.T - Y_test.T) / len(result))

# Prepare graph
blue = []
red = []
for i in range(len(result)):
    if Y_test[i] == 1:
        red.append(X_test[i])
    else:
        blue.append(X_test[i])

fig = plt.figure()
ax = fig.add_subplot(111)

# Graph our true values
ax.scatter([i[0] for i in red], [j[1] for j in red], s=25, c='r', marker='o', label='Virginica')
ax.scatter([i[0] for i in blue], [j[1] for j in blue], s=25, c='b', marker='s', label='Not Virginica')

# Plot the decision boundary
slope = -model[0] / model[1]
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = slope * x_vals
plt.plot(x_vals, y_vals, '--')

# Actually plot it
plt.legend(loc='upper right');
ax.set_title("Plot hw1")
ax.set_xlabel('Sepal Height')
ax.set_ylabel('Sepal Width')
plt.show()