# example of plotting a gradient descent search on a one-dimensional function
from numpy import asarray
from numpy import arange
from numpy.random import rand
from matplotlib import pyplot
import numpy as np


# objective function
def objective(x):
    return x ** 2.0


# derivative of objective function
def derivative(x):
    return x * 2.0


# gradient descent algorithm
def gradient_descent(objective, derivative, bounds, n_iter, step_size):
    # track all solutions
    solutions, scores = list(), list()
    # generate an initial point
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # run the gradient descent
    for i in range(n_iter):
        # calculate gradient
        gradient = derivative(solution)
        # take a step
        solution = solution - step_size * gradient
        # evaluate candidate point
        solution_eval = objective(solution)
        # store solution
        solutions.append(solution)
        scores.append(solution_eval)
        # report progress
        print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
    return [solutions, scores]


def LinearCG(A, b, x0, tol=1e-5):
	xk = x0
	rk = np.dot(A, xk) - b
	pk = -rk
	rk_norm = np.linalg.norm(rk)

	num_iter = 0
	curve_x = [xk]
	while rk_norm > tol:
		apk = np.dot(A, pk)
		rkrk = np.dot(rk, rk)

		alpha = rkrk / np.dot(pk, apk)
		xk = xk + alpha * pk
		rk = rk + alpha * apk
		beta = np.dot(rk, rk) / rkrk
		pk = -rk + beta * pk

		num_iter += 1
		curve_x.append(xk)
		rk_norm = np.linalg.norm(rk)
		print('Iteration: {} \t x = {} \t residual = {:.4f}'.
			  format(num_iter, xk, rk_norm))

	print('\nSolution: \t x = {}'.format(xk))

	return np.array(curve_x)

# define range for input
bounds = asarray([[-1.0, 1.0]])
# define the total iterations
n_iter = 30
# define the step size
step_size = 0.1
# perform the gradient descent search
solutions, scores = gradient_descent(objective, derivative, bounds, n_iter, step_size)
# sample input range uniformly at 0.1 increments
inputs = arange(bounds[0, 0], bounds[0, 1] + 0.1, 0.1)
# compute targets
results = objective(inputs)
# create a line plot of input vs result
pyplot.plot(inputs, results)
# plot the solutions found
pyplot.plot(solutions, scores, '.-', color='red')
# show the plot
pyplot.show()