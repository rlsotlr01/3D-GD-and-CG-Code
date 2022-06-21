import numpy as np

def f(A,b,x):
    return 0.5*np.dot(np.dot(x,A),x)-np.dot(x,b)

def df(A,b,x):
    return np.dot(A,x)-b


def gradient_descent(A, b, x0):
    # initial guess vector x
    next_x = x0 * np.ones(n)
    # print initial f value
    print('i = 0 ; f(x)= ' + str(f(A, b, next_x)))
    i = 1
    # convergence flag
    cvg = False
    print('Starting descent')
    while i <= max_iters:
        curr_x = next_x
        next_x = curr_x - gamma * df(A, b, curr_x)
        step = next_x - curr_x
        # convergence test
        if np.linalg.norm(step, 2) / (np.linalg.norm(next_x, 2) + np.finfo(float).eps) <= tol:
            cvg = True
            break
        # print optionnaly f values while searching for minimum
        print('i = ' + str(i) + ' ; f(x)= ' + str(f(A, b, next_x)))
        i += 1
    if cvg:
        print('Minimum found in ' + str(i) + ' iterations.')
        print('x_sol =', next_x)
    else:
        print('No convergence for specified parameters.')

    return next_x

if __name__ == '__main__':
    # int: you chose 6 as initial guess
    # gradient descent parameters
    gamma = 0.01  # step size multiplier
    tol = 1e-5  # convergence tolerance for stopping criterion
    max_iters = 1e6  # maximum number of iterations

    # dimension of the problem
    n = 10

    A = np.diag(np.ones(n - 2), k=-2) + np.diag(-4 * np.ones(n - 1), k=-1) + \
        np.diag(6 * np.ones(n), k=0) + \
        np.diag(-4 * np.ones(n - 1), k=1) + np.diag(np.ones(n - 2), k=2)

    b = np.zeros(n)
    b[0] = 3;
    b[1] = -1;
    b[n - 1] = 3;
    b[n - 2] = -1
    x0 = 6

    x_sol = gradient_descent(A, b, x0)
