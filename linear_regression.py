import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# read in data
data = pd.read_csv("C:/Users/na1r_/Downloads/fish_data.csv")

# store features in X, the ground truth for the weight of the fish in y
X = data.iloc[:, 2:]
y = data['Weight']

# convert to numpy arrays
X = np.asarray(X)
y = np.asarray(y).reshape(X.shape[0], 1)

def normalize(X):
    '''
    Normalizes features in X by subtracting column-wise mean and dividing
    by column-wise standard deviation.
    '''
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = np.zeros_like(X)

    for i in range(X.shape[1]):
        X_norm[:, i] = (X[:, i] - mu[i]) / sigma[i]

    return X_norm, mu, sigma

def addOnes(X):
    '''
    Adds bias terms to the feature matrix
    '''
    onesX = np.hstack(( (np.ones((X.shape[0], 1))), X))

    return onesX

def computeCost(X, y, theta):
    '''
    Given the features, the ground truth, and parameters theta,
    compute the mean squared error.
    '''
    m = len(y)
    
    J = (1 / (2*m)) * ( ((X@theta - y).T) @ (X@theta - y))
    
    return np.squeeze(J)

def gradientDescent(X, y, theta, alpha, iterations):
    '''
    Perform gradient descent on the inputted theta given
    a set learning rate alpha and a set number of iterations.
    '''
    m = len(y)
    optTheta = theta
    costs = []
    
    for iter in range(iterations):
        
        grad = (1/m) * (X.T @ (X@optTheta - y))
        optTheta = optTheta - alpha * grad
        cost = computeCost(X, y, optTheta)
        costs.append(cost)

        # print cost every 50 iterations
        if iter % 50 == 0:
            print("Cost after iteration " + str(iter) + ": " + str(cost))
        
    return optTheta, costs

def linearRegression(X, y, alpha, iterations):
    '''
    This model calls all the necessary commands to perform
    multivariate linear regression.
    '''
    # normalize X and add biases
    X_norm, mu, sigma = normalize(X)
    X_norm_biases = addOnes(X_norm)

    print("X shape: " + str(X_norm_biases.shape))
    print("y shape: " + str(y.shape))

    # initialize and optimize theta
    theta = np.zeros((X_norm_biases.shape[1], 1))
    theta, costs = gradientDescent(X_norm_biases, y, theta, alpha, iterations)

    # plot costs
    plt.plot(costs)
    plt.show()
    
    return theta, mu, sigma

def normalEq(X, y):
    '''
    Uses the normal equation to analyticially compute
    the optimal theta.
    '''
    theta = np.linalg.pinv( (X.T) @ X ) @ (X.T @ y)

    return theta

theta, mu, sigma = linearRegression(X, y, alpha=0.1, iterations=300)
theta_normal = normalEq(X, y)

# score the models using their R2 score
X_normal, m, s = normalize(X)
h_x = addOnes(X_normal) @ theta
print("\nGradient descent R2 Score: " + str(r2_score(y, h_x)))

h_x_normal = X @ theta_normal
print("Normal equation R2 score: " + str(r2_score(y, h_x_normal)))

# make predictions on new examples
print("\n------------------")
print("Make predictions: ")

# specify values for features below
prov_X = np.array((23.2, 26.3, 30, 12.48, 4.02))

# preprocessing
prov_X_n = (prov_X - mu) / sigma
prov_X_n = np.hstack((1, prov_X_n))

# compute the estimated value
user_h_x = prov_X_n @ theta
print("Predicted fish weight given features: " + str(np.squeeze(user_h_x)))









    
        
        



















