import numpy as np
from warnings import warn
from matplotlib import pyplot as plt

from matplotlib.ticker import MaxNLocator
from itertools import product
from collections import defaultdict
import numpy as np

import time


# TO-DO-LIST : Bayesian optimization 을 넣으면 어떨까?
def WolfeLineSearch(f, f_grad, xk, pk, c1=1e-4, c2=0.9, amax=None, maxiter=10):
    """
    Find alpha that satisfies strong Wolfe conditions.
    Parameters
    ----------
    f : callable f(x)
        Objective function.
    f_grad : callable f'(x)
        Objective function gradient.
    xk : ndarray
        Starting point.
    pk : ndarray
        Search direction.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size
    maxiter : int, optional
        Maximum number of iterations to perform.
    Returns
    -------
    alpha : float or None
        Alpha for which ``x_new = x0 + alpha * pk``,
        or None if the line search algorithm did not converge.
    phi : float or None
        New function value ``f(x_new)=f(x0+alpha*pk)``,
        or None if the line search algorithm did not converge.
    """

    def phi(alpha):
        return f(xk + alpha * pk)

    def derphi(alpha):
        return np.dot(f_grad(xk + alpha * pk), pk)

    alpha_star, phi_star, derphi_star = WolfeLineSearch2(phi, derphi, c1, c2, amax, maxiter)

    # ** 0528 revise

    # if derphi_star is None:
    #     warn('The line search algorithm did not converge', RuntimeWarning)

    return alpha_star, phi_star


def WolfeLineSearch2(phi, derphi, c1=1e-4, c2=0.9, amax=None, maxiter=10):
    """
    Find alpha that satisfies strong Wolfe conditions.
    alpha > 0 is assumed to be a descent direction.
    Parameters
    ----------
    phi : callable phi(alpha)
        Objective scalar function.
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size.
    maxiter : int, optional
        Maximum number of iterations to perform.
    Returns
    -------
    alpha_star : float or None
        Best alpha, or None if the line search algorithm did not converge.
    phi_star : float
        phi at alpha_star.
    derphi_star : float or None
        derphi at alpha_star, or None if the line search algorithm
        did not converge.
    """

    phi0 = phi(0.)
    derphi0 = derphi(0.)

    alpha0 = 0
    alpha1 = 1.0

    if amax is not None:
        alpha1 = min(alpha1, amax)

    phi_a1 = phi(alpha1)
    # derphi_a1 = derphi(alpha1) evaluated below

    phi_a0 = phi0
    derphi_a0 = derphi0

    for i in range(maxiter):
        if alpha1 == 0 or (amax is not None and alpha0 == amax):
            # alpha1 == 0: This shouldn't happen. Perhaps the increment has
            # slipped below machine precision?
            alpha_star = None
            phi_star = phi0
            derphi_star = None

            if alpha1 == 0:
                msg = 'Rounding errors prevent the line search from converging'
            else:
                msg = "The line search algorithm could not find a solution " + \
                      "less than or equal to amax: %s" % amax

            warn(msg, RuntimeWarning)
            break

        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or \
                ((phi_a1 >= phi_a0) and (i > 1)):
            alpha_star, phi_star, derphi_star = \
                _zoom(alpha0, alpha1, phi_a0,
                      phi_a1, derphi_a0, phi, derphi,
                      phi0, derphi0, c1, c2)
            break

        derphi_a1 = derphi(alpha1)
        if (abs(derphi_a1) <= -c2 * derphi0):
            alpha_star = alpha1
            phi_star = phi_a1
            derphi_star = derphi_a1
            break

        if (derphi_a1 >= 0):
            alpha_star, phi_star, derphi_star = \
                _zoom(alpha1, alpha0, phi_a1,
                      phi_a0, derphi_a1, phi, derphi,
                      phi0, derphi0, c1, c2)
            break

        alpha2 = 2 * alpha1  # increase by factor of two on each iteration
        if amax is not None:
            alpha2 = min(alpha2, amax)
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1

    else:
        # stopping test maxiter reached
        alpha_star = alpha1
        phi_star = phi_a1

        # ** 0528 수정
        # derphi_star = None
        # warn('The line search algorithm did not converge', RuntimeWarning)

        # ** 0528 add
        derphi_star = 0
        # warn('The line search algorithm did not converge', RuntimeWarning)

    return alpha_star, phi_star, derphi_star


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
    If no minimizer can be found, return None.
    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                            fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa.
    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
          phi, derphi, phi0, derphi0, c1, c2):
    """
    Zoom stage of approximate linesearch satisfying strong Wolfe conditions.
    """

    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while True:
        # interpolate to find a trial step length between a_lo and
        # a_hi Need to choose interpolation here. Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by a_lo or a_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection

        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi

        # minimizer of cubic interpolant
        # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval), then use quadratic interpolation with phi_lo,
        # derphi_lo and phi_hi if the result is still too close to the
        # end points (or out of the interval) then use bisection

        if (i > 0):
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                            a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b - qchk) or (a_j < a + qchk):
                a_j = a_lo + 0.5 * dalpha

        # Check new value of a_j

        phi_aj = phi(a_j)
        if (phi_aj > phi0 + c1 * a_j * derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            if abs(derphi_aj) <= -c2 * derphi0:
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj * (a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if (i > maxiter):
            # Failed to find a conforming step size
            a_star = None
            val_star = None
            valprime_star = None
            break
    return a_star, val_star, valprime_star


# https://jamesmccaffrey.wordpress.com/2021/10/11/graphing-the-michalewicz-function-using-matplotlib/
# michalewicz_graph.py source code

def Michalewicz(xs):
    x=xs[0]
    y=xs[1]
    z = -1 * ((np.sin(x) * np.sin((1 * x ** 2) / np.pi) ** 20) + \
              (np.sin(y) * np.sin((2 * y ** 2) / np.pi) ** 20))
    return z

def MichalewiczGrad(xs):
    '''Find the Gradient of Michalewicz Function with the finite method'''
    x = xs[0]
    y = xs[1]
    x_grad = -(1/np.pi)*((np.sin(x**2/np.pi))**19 * (40*x*np.sin(x)*np.cos(x**2/np.pi) + np.pi*np.sin(x**2/np.pi)*np.cos(x)))
    y_grad = -(1/np.pi)*((np.sin(2*y**2/np.pi))**19)*(80*y*np.sin(y)*np.cos(2*y**2/np.pi) + np.pi*np.sin(2*y**2/np.pi)*np.cos(y))
    return np.array([x_grad, y_grad])


def MichalewiczGrad2(xs):
    '''Find the Gradient of Michalewicz Function with the finite method'''
    x = xs[0]
    y = xs[1]
    interval = 0.0001
    x_grad = (Michalewicz(np.array([x + interval, y])) - Michalewicz(np.array([x, y]))) / interval
    y_grad = (Michalewicz(np.array([x, y+ interval])) - Michalewicz(np.array([x, y]))) / interval
    return np.array([x_grad, y_grad])

def Griewank(xs):
    """Griewank Function"""
    d = len(xs)
    sqrts = np.array([np.sqrt(i + 1) for i in range(d)])
    cos_terms = np.cos(xs / sqrts)

    sigma = np.dot(xs, xs) / 4000
    pi = np.prod(cos_terms)
    return 1 + sigma - pi


def GriewankGrad(xs):
    """First derivative of Griewank Function"""
    d = len(xs)
    sqrts = np.array([np.sqrt(i + 1) for i in range(d)])
    cos_terms = np.cos(xs / sqrts)
    pi_coefs = np.prod(cos_terms) / cos_terms

    sigma = 2 * xs / 4000
    pi = pi_coefs * np.sin(xs / sqrts) * (1 / sqrts)
    return sigma + pi


def NonlinearCG(f, f_grad, init, method='FR', c1=1e-4, c2=0.1, amax=None, tol=1e-5, max_iter=1000):
    """Non Linear Conjugate Gradient Method for optimization problem.
    Given a starting point x ∈ ℝⁿ.
    repeat
        1. Calculate step length alpha using Wolfe Line Search.
        2. Update x_new = x + alpha * p.
        3. Calculate beta using one of available methods.
        4. Update p = -f_grad(x_new) + beta * p
    until stopping criterion is satisfied.

    Parameters
    --------------------
        f        : function to optimize
        f_grad   : first derivative of f
        init     : initial value of x, can be set to be any numpy vector,
        method   : method to calculate beta, can be one of the followings: FR, PR, HS, DY, HZ.
        c1       : Armijo constant
        c2       : Wolfe constant
        amax     : maximum step size
        tol      : tolerance of the difference of the gradient norm to zero
        max_iter : maximum number of iterations

    Returns
    --------------------
        curve_x  : x in the learning path
        curve_y  : f(x) in the learning path
    """

    # initialize some values
    x = init
    y = f(x)
    gfk = f_grad(x)
    offset_interval = 0.1*np.ones(2)
    # offset_interval = offset_interval*np.ones(2)
    # TODO : Offset Method
    # ****** Offset Method
    # while gfk.mean() <= 1e-4:
    #     x = init+offset_interval
    #     y = f(x)
    #     gfk = f_grad(x)

    p = -gfk
    gfk_norm = np.linalg.norm(gfk)

    # for result tabulation
    num_iter = 0
    curve_x = [x]
    curve_y = [y]
    print('Initial condition: y = {:.4f}, x = {} \n'.format(y, x))

    # begin iteration
    while gfk_norm > tol and num_iter < max_iter:
        # search for step size alpha
        alpha, y_new = WolfeLineSearch(f, f_grad, x, p, c1=c1, c2=c2, amax=amax)


        # ** 0528 revise
        if alpha == None:
            break

        # update iterate x
        x_new = x + alpha * p
        gf_new = f_grad(x_new)

        # calculate beta
        if method == 'FR':
            beta = np.dot(gf_new, gf_new) / np.dot(gfk, gfk)
        elif method == 'PR':
            y_hat = gf_new - gfk
            beta = np.dot(gf_new, y_hat) / np.dot(gfk, gfk)
        elif method == 'HS':
            y_hat = gf_new - gfk
            beta = np.dot(y_hat, gf_new) / np.dot(y_hat, p)
        elif method == 'DY':
            y_hat = gf_new - gfk
            beta = np.dot(gf_new, gf_new) / np.dot(y_hat, p)
        elif method == 'HZ':
            y_hat = gf_new - gfk
            beta = np.dot(y_hat, gf_new) / np.dot(y_hat, p)
            beta = beta - 2 * np.dot(y_hat, y_hat) * np.dot(p, gf_new) / (np.dot(y_hat, p) ** 2)
        else:
            raise ValueError(
                'Method is unrecognizable. Try one of the following values: FR, PR, HS, DY, HZ.'
            )

        # update everything
        error = y - y_new
        x = x_new
        y = y_new
        gfk = gf_new
        p = -gfk + beta * p
        gfk_norm = np.linalg.norm(gfk)

        # result tabulation
        num_iter += 1
        curve_x.append(x)
        curve_y.append(y)
        print('Iteration: {} \t y = {:.4f}, x = {}, gradient = {:.4f}'.
              format(num_iter, y, x, gfk_norm))

    # print results
    if num_iter == max_iter:
        print('\nGradient descent does not converge.')
    else:
        print('\nSolution: \t y = {:.4f}, x = {}'.format(y, x))

    return np.array(curve_x), np.array(curve_y)


def ArmijoLineSearch(f, xk, pk, gfk, phi0, alpha0, rho=0.5, c1=1e-4):
    """Minimize over alpha, the function ``f(xₖ + αpₖ)``.
    α > 0 is assumed to be a descent direction.

    Parameters
    --------------------
    f : callable
        Function to be minimized.
    xk : array
        Current point.
    pk : array
        Search direction.
    gfk : array
        Gradient of `f` at point `xk`.
    phi0 : float
        Value of `f` at point `xk`.
    alpha0 : scalar
        Value of `alpha` at the start of the optimization.
    rho : float, optional
        Value of alpha shrinkage factor.
    c1 : float, optional
        Value to control stopping criterion.

    Returns
    --------------------
    alpha : scalar
        Value of `alpha` at the end of the optimization.
    phi : float
        Value of `f` at the new point `x_{k+1}`.
    """
    derphi0 = np.dot(gfk, pk)
    phi_a0 = f(xk + alpha0 * pk)

    while not phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        alpha0 = alpha0 * rho
        phi_a0 = f(xk + alpha0 * pk)

    return alpha0, phi_a0

def GradientDescent(f, f_grad, init, alpha=1, tol=1e-5, max_iter=1000):
    """Gradient descent method for unconstraint optimization problem.
    given a starting point x ∈ Rⁿ,
    repeat
        1. Define direction. p := −∇f(x).
        2. Line search. Choose step length α using Armijo Line Search.
        3. Update. x := x + αp.
    until stopping criterion is satisfied.

    Parameters
    --------------------
    f : callable
        Function to be minimized.
    f_grad : callable
        The first derivative of f.
    init : array
        initial value of x.
    alpha : scalar, optional
        the initial value of steplength.
    tol : float, optional
        tolerance for the norm of f_grad.
    max_iter : integer, optional
        maximum number of steps.

    Returns
    --------------------
    xs : array
        x in the learning path
    ys : array
        f(x) in the learning path
    """
    # initialize x, f(x), and f'(x)
    xk = init
    fk = f(xk)
    gfk = f_grad(xk)
    gfk_norm = np.linalg.norm(gfk)
    # initialize number of steps, save x and f(x)
    num_iter = 0
    curve_x = [xk]
    curve_y = [fk]
    print('Initial condition: y = {:.4f}, x = {} \n'.format(fk, xk))
    # take steps
    while gfk_norm > tol and num_iter < max_iter:
        # determine direction
        pk = -gfk
        # calculate new x, f(x), and f'(x)
        alpha, fk = ArmijoLineSearch(f, xk, pk, gfk, fk, alpha0=alpha)
        xk = xk + alpha * pk
        gfk = f_grad(xk)
        gfk_norm = np.linalg.norm(gfk)
        # increase number of steps by 1, save new x and f(x)
        num_iter += 1
        curve_x.append(xk)
        curve_y.append(fk)
        print('Iteration: {} \t y = {:.4f}, x = {}, gradient = {:.4f}'.
              format(num_iter, fk, xk, gfk_norm))
    # print results
    if num_iter == max_iter:
        print('\nGradient descent does not converge.')
    else:
        print('\nSolution: \t y = {:.4f}, x = {}'.format(fk, xk))

    return np.array(curve_x), np.array(curve_y)

def plot(xs, ys, title, x_min=-5, x_max=5, y_min=-5, y_max=5):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle('Conjugate Gradient Method')
    interval = abs(x_max)/200.
    x = np.arange(x_min, x_max, interval)
    y = np.arange(y_min, y_max, interval)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    mesh_size = range(len(X))
    for i, j in product(mesh_size, mesh_size):
        x_coor = X[i][j]
        y_coor = Y[i][j]
        Z[i][j] = Griewank(np.array([x_coor, y_coor]))

    ax1.plot(xs[:,0], xs[:,1], linestyle='--', marker='o', color='orange')
    ax1.plot(xs[-1,0], xs[-1,1], 'ro')
    ax1.set(
        title=title,
        xlabel='x1',
        ylabel='x2'
    )
    CS = ax1.contour(X, Y, Z)
    ax1.clabel(CS, fontsize='smaller', fmt='%1.2f')
    ax1.axis('square')

    ax2.plot(ys, linestyle='--', marker='o', color='orange')
    ax2.plot(len(ys)-1, ys[-1], 'ro')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set(
        title='Objective Function Value During Optimization Process',
        xlabel='Iterations',
        ylabel='Objective Function Value'
    )
    ax2.legend(['Armijo line search algorithm'])

    plt.tight_layout()
    plt.show()

def plot_xss(xss, title, x_min=-5, x_max=5, y_min=-5, y_max=5):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    # plt.suptitle('Gradient Descent Method')
    print('x_min >>>>>', x_min)
    interval = abs(x_max)/200.
    x = np.arange(x_min*2, x_max*2, interval)
    y = np.arange(x_min*2, x_max*2, interval)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    mesh_size = range(len(X))
    for i, j in product(mesh_size, mesh_size):
        x_coor = X[i][j]
        y_coor = Y[i][j]
        Z[i][j] = Griewank(np.array([x_coor, y_coor]))
    for key in xss.keys():
        ax1.plot(xss[key][0][:,0],xss[key][0][:,1], linestyle='--', marker='o')
        ax1.plot(xss[key][0][-1,0],xss[key][0][-1,1], 'ro')
        ax1.set(
            title=title,
            xlabel='x1',
            ylabel='x2'
        )
        CS = ax1.contour(X, Y, Z)

    ax1.clabel(CS, fontsize='smaller', fmt='%1.2f')
    ax1.axis('square')

    # *** 0528 legend delete
    # ax1.legend(xss.keys())

    # ax2.plot(ys, linestyle='--', marker='o', color='orange')
    # ax2.plot(len(ys)-1, ys[-1], 'ro')
    # ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax2.set(
    #     title='Objective Function Value During Optimization Process',
    #     xlabel='Iterations',
    #     ylabel='Objective Function Value'
    # )
    # ax2.legend(['Armijo line search algorithm'])

    plt.tight_layout()
    plt.show()

def plot_3d_griewank(x_min=-5, x_max=5, y_min=-5, y_max=5):
    interval = abs(x_max)/200.
    x = np.arange(x_min, x_max, interval)
    y = np.arange(x_min, x_max, interval)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    mesh_size = range(len(X))
    for i, j in product(mesh_size, mesh_size):
        x_coor = X[i][j]
        y_coor = Y[i][j]
        Z[i][j] = Griewank(np.array([x_coor, y_coor]))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.set_title('3D Griewank Function')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1, x_2)$')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.tight_layout()
    plt.show()



def plot_michalewicz(xs, ys, title, x_min=-5, x_max=5, y_min=-5, y_max=5):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # plt.suptitle('Gradient Descent Method')
    interval = abs(x_max)/200.
    x = np.arange(x_min, x_max, interval)
    y = np.arange(y_min, y_max, interval)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    mesh_size = range(len(X))
    for i, j in product(mesh_size, mesh_size):
        x_coor = X[i][j]
        y_coor = Y[i][j]
        Z[i][j] = Michalewicz(np.array([x_coor, y_coor]))

    ax1.plot(xs[:,0], xs[:,1], linestyle='--', marker='o', color='orange')
    ax1.plot(xs[-1,0], xs[-1,1], 'ro')
    ax1.set(
        title=title,
        xlabel='x1',
        ylabel='x2'
    )
    CS = ax1.contour(X, Y, Z)
    ax1.clabel(CS, fontsize='smaller', fmt='%1.2f')
    ax1.axis('square')

    ax2.plot(ys, linestyle='--', marker='o', color='orange')
    ax2.plot(len(ys)-1, ys[-1], 'ro')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set(
        title='Objective Function Value During Optimization Process',
        xlabel='Iterations',
        ylabel='Objective Function Value'
    )
    ax2.legend(['Armijo line search algorithm'])

    plt.tight_layout()
    plt.show()

def plot_xss_michalewicz(xss, title, x_min=-5, x_max=5, y_min=-5, y_max=5):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    plt.suptitle('Gradient Descent Method')
    print('x_min >>>>>', x_min)
    interval = abs(x_max)/200.
    x = np.arange(x_min, x_max, interval)
    y = np.arange(x_min, x_max, interval)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    mesh_size = range(len(X))
    for i, j in product(mesh_size, mesh_size):
        x_coor = X[i][j]
        y_coor = Y[i][j]
        Z[i][j] = Michalewicz(np.array([x_coor, y_coor]))
    for key in xss.keys():
        ax1.plot(xss[key][0][:,0],xss[key][0][:,1], linestyle='--', marker='o')
        ax1.plot(xss[key][0][-1,0],xss[key][0][-1,1], 'ro')
        ax1.set(
            title=title,
            xlabel='x1',
            ylabel='x2'
        )
        CS = ax1.contour(X, Y, Z)

    ax1.clabel(CS, fontsize='smaller', fmt='%1.2f')
    ax1.axis('square')

    # 0528 legend delete
    # ax1.legend(xss.keys())

    # ax2.plot(ys, linestyle='--', marker='o', color='orange')
    # ax2.plot(len(ys)-1, ys[-1], 'ro')
    # ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax2.set(
    #     title='Objective Function Value During Optimization Process',
    #     xlabel='Iterations',
    #     ylabel='Objective Function Value'
    # )
    # ax2.legend(['Armijo line search algorithm'])

    plt.tight_layout()
    plt.show()

def plot_3d_michalewicz(x_min=-5, x_max=5, y_min=-5, y_max=5):
    interval = abs(x_max)/200.
    x = np.arange(x_min, x_max, interval)
    y = np.arange(x_min, x_max, interval)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    mesh_size = range(len(X))
    for i, j in product(mesh_size, mesh_size):
        x_coor = X[i][j]
        y_coor = Y[i][j]
        Z[i][j] = Michalewicz(np.array([x_coor, y_coor]))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.set_title('3D Michalewicz Function')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1, x_2)$')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.tight_layout()
    plt.show()


def func1(x1, x2):
    return 100*(x2-x1**2)**2

def func2(x1, x2):
    return (1-x1)**2

def func1_2(x1,x2):
    return 0.5*(func1(x1,x2)**2+func2(x1,x2)**2)


def GlobalNonlinearCG(func, funcGrad, method='DY', min_x = -20
                      , max_x = 20, min_y = -20, max_y = 20, max_trial=1000, max_iter=100):
# **** Global optimal Algorithm
# Decide the range for searching (min_x, max_x, min_y, max_y),
# Decide the maximum trial to find the global optimum by entering the initial point

# if -20~20 으로 범위 변경하면 경사 0으로 인한 에러가 생김.
# 해결해야될 듯
    count = 0
    min_ys = 10



    trial_num = 0
    interval = 1
    x1 = np.arange(min_x, max_x, interval)
    x2 = np.arange(min_y, max_y, interval)
    # x1 = range(min_x,max_x,1)
    # [0, 1, 2, 3, 4]
    # x2 = range(min_y,max_y,1)
    # [0, 1, 2, 3, 4]
    xss = defaultdict(list)
    keys = []
    min_count = defaultdict(int)
    for i in x1:
        for j in x2:
            key = str(i)+str('-')+str(j)
            print('>>>>>>>>>>>>>> key: ',key)
            xss[key] = []
            x0 = np.array([i, j])
            xs, ys = NonlinearCG(func, funcGrad, init=x0, method='HS', max_iter=100)
            xss[key].append(xs)
            xss[key].append(ys)
            #
            # if min_ys-min(ys) <= 1e-4:
            #     key = str(round(min_ys,4))
            #     min_count[key] += 1

            if min_ys >= min(ys):
                key = str(round(min(ys), 4))
                min_count[key] += 1
                min_ys = min(ys)
            if min_ys >= min(ys):
                trial_num += 1

            count += 1
            min_ys = min(ys)
            print(">>>>>>>>> min_ys: ", min_ys)
            print(">>>>>>>>> count: ", count)
            print(">>>>>>>>> min_count: ", min_count)
            if trial_num >= max_trial:
                print("the trial num is over")
                break

    # Find out the suboptimal point that it found.
    title = 'NonlinearCG'
    plot_xss(xss, title, x_min=min_x-1, x_max=max_x+1, y_min=min_y-1, y_max=max_y+1)
    print(">>>>>>>>> Trial number:", trial_num)
    # print('>>>>>>>>> trial_num: ', trial_num)
    print('>>>>>>>>>> Global minimum:',min_ys)
    pass



def GlobalNonlinearCG_michal(func, funcGrad, method='DY', min_x = -20
                      , max_x = 20, min_y = -20, max_y = 20, max_trial=1000, max_iter=100):
# **** Global optimal Algorithm
# Decide the range for searching (min_x, max_x, min_y, max_y),
# Decide the maximum trial to find the global optimum by entering the initial point

# if -20~20 으로 범위 변경하면 경사 0으로 인한 에러가 생김.
# 해결해야될 듯
    count = 0
    min_ys = 10
    trial_num = 0

    interval = 0.5
    x1 = np.arange(min_x, max_x, interval)
    x2 = np.arange(min_y, max_y, interval)
    # x1 = range(min_x-3,max_x+3,0.1)
    # [0, 1, 2, 3, 4]
    # x2 = range(min_y-3,max_y+3,0.1)
    # [0, 1, 2, 3, 4]
    xss = defaultdict(list)
    keys = []
    min_count = defaultdict(int)
    for i in x1:
        for j in x2:
            key = str(i)+str('-')+str(j)
            print('>>>>>>>>>>>>>> key: ',key)
            xss[key] = []

            # 0528 revise - add noise
            noise = np.random.rand()/10

            # np.array([i,j]) -> np.array([i+noise, j+noise])
            x0 = np.array([i, j])
            xs, ys = NonlinearCG(func, funcGrad, init=x0, method='FR', max_iter=100)
            xss[key].append(xs)
            xss[key].append(ys)

            # if min_ys-min(ys) <= 1e-4:
            #     key = str(round(min_ys,4))
            #     min_count[key] += 1
            if min_ys >= min(ys):
                key = str(round(min(ys),4))
                min_count[key] += 1
                min_ys = min(ys)
            if min_ys >= min(ys):
                trial_num += 1

            count += 1
            # min_ys = min(ys)
            print(">>>>>>>>> min_ys: ", min_ys)
            print(">>>>>>>>> count: ", count)
            print(">>>>>>>>> Trial number:", trial_num)
            print(">>>>>>>>> min_count: ", min_count)
            if trial_num >= max_trial:
                print("the trial num is over")
                break

    # Find out the suboptimal point that it found.
    title = 'NonlinearCG'
    plot_xss_michalewicz(xss, title, x_min=min_x-3, x_max=max_x+3, y_min=min_y-3, y_max=max_y+3)
    print('>>>>>>>>> trial_num: ', trial_num)
    print('>>>>>>>>>> Global minimum:',min_ys)
    pass

if __name__ == '__main2__':
    x0 = np.array([2, 1])
    xs, ys = NonlinearCG(Griewank, GriewankGrad, init=x0, method='DY')
    plot(xs, ys, 'alkslds')

if __name__ == '__main__':

    #
    # x0 = np.array([0, 3])
    # xs, ys = GradientDescent(Griewank, GriewankGrad, init=x0)
    # title = 'Gradient Descent'
    # plot(xs, ys, title)
    #
    # x0 = np.array([0, 3])
    # xs, ys = NonlinearCG(Griewank, GriewankGrad, init=x0, method='FR')
    # title = 'Nonlinear CG with FR'
    # plot(xs, ys, title)
    # print('xs\n', xs)


    # plot_3d_griewank(x_min=-5, x_max=5, y_min=-5, y_max=5)
    # plot_3d_michalewicz(x_min=0, x_max=3, y_min=0, y_max=3)

    # ***Michalewicz Gradient Descent - Success

    # x0 = np.array([2, 1])
    # xs, ys = GradientDescent(Michalewicz, MichalewiczGrad, init=x0)
    # title = 'GD with Michalewicz'
    # plot_michalewicz(xs, ys, title, x_min=-1, x_max=3, y_min=-1, y_max=3)
    #
    # x0 = np.array([2.1, 1.5])
    # xs, ys = NonlinearCG(Michalewicz, MichalewiczGrad, init=x0)
    # title = 'CG with Michalewicz'
    # plot_michalewicz(xs, ys, title, x_min=1, x_max=3, y_min=1, y_max=3)

    # start = time.time()
    # GlobalNonlinearCG(Griewank, GriewankGrad, method='DY', min_x=-5, max_x=5, min_y=-5, max_y=5)
    # end = time.time()
    # time_griewank = end - start

    # print('>>>>>>>Spending Time for global method:', end - start)

    # Check the CG can find the optimal point of Michalewicz function's optimal solution
    # x0 = np.array([2, 1])
    # xs, ys = NonlinearCG(Michalewicz, MichalewiczGrad, init=x0)
    # title = 'CG with Michalewicz'
    # plot_michalewicz(xs, ys, title, x_min=-1, x_max=3, y_min=-1, y_max=3)
    #
    start = time.time()
    GlobalNonlinearCG_michal(Michalewicz, MichalewiczGrad, method='DY', min_x=0, max_x=3, min_y=0, max_y=3)
    end = time.time()
    #
    print('>>>>>>>Spending Time for global method with Michalewicz:',end-start)
    # print('>>>>>>>Spending Time for global method with Griewank:', time_griewank)
    # GlobalNonlinearCG_michal(Michalewicz, MichalewiczGrad, method='DY', min_x=0, max_x=5, min_y=0, max_y=5)


    # *** Gradient Descent for Griewank
    # x0 = np.array([2, 1])
    # xs, ys = GradientDescent(Griewank, GriewankGrad, init=x0)
    # title = 'Gradient Descent'
    # plot(xs, ys, title)


    # x0 = np.array([1, 2])
    # xs, ys = GradientDescent(Griewank, GriewankGrad, init=x0)
    # title = 'Gradient Descent'
    # plot(xs, ys, title)
    '''
    start = time.time()
    x0 = np.array([2, 1])
    xs, ys = GradientDescent(Griewank, GriewankGrad, init=x0)
    title = 'Gradient Descent'
    plot(xs, ys, title)
    end = time.time()
    print('spending time for GD: ', end-start)

    start = time.time()
    x0 = np.array([2, 1])
    xs, ys = NonlinearCG(Griewank, GriewankGrad, init=x0, method='HS')
    title = 'Nonlinear CG'
    plot(xs, ys, title)
    end = time.time()
    print('spending time for CG: ', end - start)
    '''
    '''
    x0 = np.array([2, 1])
    xs, ys = NonlinearCG(Griewank, GriewankGrad, init=x0, method='PR')
    title = 'Nonlinear CG with PR'
    plot(xs, ys, title)

    x0 = np.array([2, 1])
    xs, ys = NonlinearCG(Griewank, GriewankGrad, init=x0, method='HS')
    title = 'Nonlinear CG with HS'
    plot(xs, ys, title)

    x0 = np.array([2, 1])
    xs, ys = NonlinearCG(Griewank, GriewankGrad, init=x0, method='DY')
    title = 'Nonlinear CG with DY'
    plot(xs, ys, title)

    x0 = np.array([2, 1])
    xs, ys = NonlinearCG(Griewank, GriewankGrad, init=x0, method='HZ')
    title = 'Nonlinear CG with HZ'
    plot(xs, ys, title)

    # x0 = np.array([1, 3])
    # xs, ys = GradientDescent(Griewank, GriewankGrad, init=x0)
    # title = 'Gradient Descent'
    # plot(xs, ys, title)

    x0 = np.array([2, 2])
    xs, ys = GradientDescent(Griewank, GriewankGrad, init=x0)
    title = 'Gradient Descent'
    plot(xs, ys, title)

    x0 = np.array([2, 2])
    xs, ys = NonlinearCG(Griewank, GriewankGrad, init=x0, method='HZ')
    title = 'Nonlinear CG with HZ'
    plot(xs, ys, title)


    # x0 = np.array([3, 1])
    # xs, ys = GradientDescent(Griewank, GriewankGrad, init=x0)
    # title = 'Gradient Descent'
    # plot(xs, ys, title)
    '''

