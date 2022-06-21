import numpy as np


def gradient_descent(X, y, theta, learning_rate = 0.01, iterations = 100):
    '''
    :param X: Matrix of X with added bias units
    :param y: Vector of Y
    :param theta: Vector of thetas np.random.randn(j,1)
    :param learning_rate: learning_rate
    :param iteration: no of iterations
    :return: the final theta vector and array of cost history over no of iterations
    '''
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
    for it in range(iterations):
        prediction = np.dot(X,theta)
        theta = theta - (1/m)*learning_rate*(X.T.dot((prediction-y)))
        # dJ/dX = 2*(y-prediction)
        # J = (y-prediction)**2
        theta_history[it, :] = theta.T
        cost_history[it] = cal_cost(theta,X,y)
    return theta, cost_history, theta_history

# mean cost value
def cal_cost(theta, X, y):
    '''
    :param theta: Vector of thetas
    :param X: Row of X's np.zeros((2,j))
    :param y: Actual y's np.zeros((2,1))
    :return: cost function between the predictions and real values
    where: J is the no of features - scalar
    '''
    m = len(y)
    predictions = X.dot(theta)
    return (1/(2*m))*np.sum(np.square(y-predictions))


lr = 0.01
n_iter = 1000
X = 2*np.random.rand(100,1)
y = 4+3*X+np.random.randn(100,1)

theta = np.random.randn(2,1)
X_b = np.c_[np.ones((len(X),1)),X]
theta, cost_history, theta_history = gradient_descent(X_b, y, theta, lr, n_iter)
