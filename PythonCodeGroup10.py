import numpy as np
from scipy.integrate import quad
import typing
import matplotlib.pyplot as plt
import time
from scipy.optimize import ridder

SOL_STRING = "A solution was found in {0} iterations with |a - b| = {1}, f(c) = {2}"
NO_SOL_STRING = "A solution was could not be found in {0} iterations, f(c) = {1}"

def timeit(func: typing.Callable):
    """Timeit decorator, please before a function to measure duration.

    Args:
        func (typing.Callable): Any function
    """    
    def wrapper(*arg, **kw):
        t1 = time.perf_counter()
        res = func(*arg, **kw)
        t2 = time.perf_counter()
        print("{0} took {1}s".format(func.__name__, (t2 - t1)))
        return res
    return wrapper

def funf(x:float, lmbda:float, mu:float) -> float:
    """Function for the PDF

    Args:
        x (float): Value of x
        lmbda (float): Value of lambda
        mu (float): Value of mu

    Returns:
        float: returns the function value
    """    
    if (isinstance(x, float) or isinstance(x, int)) and not x > 0:
        return 0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        y = np.where(x > 0, np.sqrt(lmbda / (2 * np.pi * x**3)) * np.exp((-lmbda / (2 * mu**2 * x)) * (x - mu)**2),0)
    return y

# PART A

def plotfunf(params: typing.Dict[float, float]):
    """Plot the function f with a list of parameters

    Args:
        params (typing.Dict[float, float]): List of parameters
    """    
    delta = 0.01
    x = np.arange(0, 10, delta)
    z = np.where(x>0,0,1)
    #print(z)
    for param in params:
        y = funf(x, **param)
        plt.plot(x, y, label=r'$\lambda={lmbda}, \mu={mu}$'.format(**param))

    plt.legend(loc="upper right")
    plt.title("Part A PDF")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()

# PART B

@timeit
def romberg(f: typing.Callable[[np.ndarray], np.ndarray], a: float, b: float, iterations: int, params: typing.Dict[float, float] = {}) -> np.ndarray:
    """Function for calculating an integral by using Romberg

    Args:
        f (typing.Callable[[np.ndarray], np.ndarray]): function (callable)
        a (float): left bound of integration
        b (float): right bound of integration
        iterations (int): amount of iterations
        params (typing.Dict[float, float], optional): parameters for the integration. Defaults to {}.

    Returns:
        np.ndarray: Romberg integration table
    """    
    table = np.zeros((iterations, iterations))

    step = b - a
    table[0, 0] = 0.5 * step * (f(a, **params) + f(b, **params))

    for i in range(1, iterations):
        step *= 0.5
        trapz_values = np.sum(f(a + np.arange(1, 2**i, 2) * step, **params))
        table[i, 0] = 0.5 * table[i-1, 0] + step * trapz_values

        # compute richardson iterations
        for rich_i in range(1, i + 1):
            table[i, rich_i] = table[i, rich_i-1] + (table[i, rich_i-1] - table[i-1, rich_i-1]) / (4**rich_i - 1)
    return table[iterations-1, iterations-1]

@timeit
def riemann(f:typing.Callable[[np.ndarray], np.ndarray], a:float, b:float, delta:float, params:typing.Dict[float, float] = {}) -> float:
    """Function for calculating an integral by using Riemann integrals

    Args:
        f (typing.Callable[[np.ndarray], np.ndarray]): function (callable)
        a (float): left bound of integration
        b (float): right bound of integration
        iterations (int): amount of iterations
        params (typing.Dict[float, float], optional): parameters for the integration. Defaults to {}.

    Returns:
        c (float): integral result
    """    
    x = np.arange(a, b, delta)
    y = f(x, **params)
    return np.sum(y * delta) 

@timeit
def integratescipy(f:typing.Callable, a:float, b:float, params:typing.Dict[float, float] = {}):
    """Function for calculating an integral by using scipy quad.

    Args:
        f (typing.Callable[[np.ndarray], np.ndarray]): function (callable)
        a (float): left bound of integration
        b (float): right bound of integration
        iterations (int): amount of iterations
        params (typing.Dict[float, float], optional): parameters for the integration. Defaults to {}.

    Returns:
        c (float): integral result
    """    
    return quad(f, a, b, (params["lmbda"], params["mu"]))

# PART C

def funl(lambdas: np.ndarray, mu: float, arr: np.ndarray) -> np.ndarray:
    """Log likelyhood function for function f

    Args:
        lambdas (np.ndarray): the list of lambdas to compute the function for
        mu (float): the parameter mu
        arr (np.ndarray): the list of x-values

    Returns:
        np.ndarray: the output log likelyhoods
    """    
    n = len(arr)
    log_likelihoods = n/2 * np.log(lambdas) - 1/2 * np.log(2*np.pi*arr**3).sum() - lambdas * (((arr - mu)**2)/((2*mu**2)*arr)).sum()
    return log_likelihoods

def funderivativel(lambdas: np.ndarray, mu: float, arr: np.ndarray) -> np.ndarray:
    """The derivative of the log-likelyhood function

    Args:
        lambdas (np.ndarray): the list of lambdas to compute the function for
        mu (float): the parameter mu
        arr (np.ndarray): the list of x-values

    Returns:
        np.ndarray: the output log likelyhoods
    """    
    n = len(arr)
    log_likelihoods_derivatives = (n/(2*lambdas)) - (((arr - mu)**2)/((2*mu**2)*arr)).sum()
    return log_likelihoods_derivatives

def plotfunl(axes, mu_hat: float, arr: np.ndarray, leftBound: float, rightBound: float, delta: float):
    """Plot the log-likelyhood function

    Args:
        axes (_type_): axes object from Matplotlib
        mu_hat (float): the value of mu_hat
        arr (np.ndarray): the array of x-values
        leftBound (float): left boundary of the graph
        rightBound (float): the right boundary of the graph
        delta (float): the distance between points

    Returns:
        _type_: Axes object
    """    
    lambda_values = np.arange(leftBound,rightBound,delta)

    log_likelihood = funl(lambda_values, mu_hat, arr)

    axes[0].plot(lambda_values, log_likelihood, label='Log Likelihood')
    axes[0].set_title(r'$\mathrm{Log\ Likelihood\ Function}$')
    axes[0].set_xlabel(r'$\lambda$')
    axes[0].set_ylabel(r'$\ell(\lambda | x)$')
    axes[0].grid(True)

    return axes[0]

def plotfunderivativel(axes, mu_hat: float, arr: np.ndarray, leftBound: float, rightBound: float, delta: float):
    """Plot the log-likelyhood function

    Args:
        axes (_type_): axes object from Matplotlib
        mu_hat (float): the value of mu_hat
        arr (np.ndarray): the array of x-values
        leftBound (float): left boundary of the graph
        rightBound (float): the right boundary of the graph
        delta (float): the distance between points

    Returns:
        _type_: Axes object
    """    
    lambda_values = np.arange(leftBound,rightBound,delta)

    derivative_log_likelihood = funderivativel(lambda_values, mu_hat, arr)

    axes[1].plot(lambda_values, derivative_log_likelihood, label='Derivative of Log Likelihood', color='orange')
    axes[1].set_title(r'$\mathrm{Derivative\ of\ Log\ Likelihood\ Function}$')
    axes[1].set_xlabel(r'$\lambda$')
    axes[1].set_ylabel(r'$\ell\'(\lambda | x)$')
    axes[1].grid(True)
    return axes[1]

# PART D

@timeit
def bisection(f: typing.Callable[[float], float], a: float, b: float, args: typing.Dict, epsilon: float=0.001) -> float:
    """Root finding method bisection

    Args:
        f (typing.Callable[[float], float]): function (callable)
        a (float): left boundary of the root finding interval
        b (float): right boundary of the root finding interval
        args (typing.Dict): function (f) arguments
        epsilon (float, optional): Minimum distance between a,b. Defaults to 0.001.

    Raises:
        Exception: This function may not converge

    Returns:
        float: The function's root
    """    
    if f(a, *args)*f(b, *args) > 0:
        raise Exception("f(a)f(b) > 0, This function may not converge")

    N_MAX = np.ceil(np.log2(np.abs(b-a)/epsilon))

    n = 0
    while n <= N_MAX:
        c = (b + a) / 2
        f_c = f(c, *args)
        
        if f_c == 0 or np.abs(b-a) < epsilon:
            print(SOL_STRING.format(n, np.abs(b-a), f_c))
            return c
        n += 1

        if f_c*f(a, *args) > 0:
            a = c
        else:
            b = c
            
    print(NO_SOL_STRING.format(n, f_c))
    return c

@timeit
def falsepos(f: typing.Callable[[float], float], a: float, b: float, args: typing.Dict, N_MAX: int=1000, epsilon: float=0.001) -> float:
    """Root finding method false position

    Args:
        f (typing.Callable[[float], float]): function (callable)
        a (float): left boundary of the root finding interval
        b (float): right boundary of the root finding interval
        args (typing.Dict): function (f) arguments
        N_MAX (int, optional): Maximum number of iterations. Defaults to 1000.
        epsilon (float, optional): Minimum distance between a,b. Defaults to 0.001.

    Raises:
        Exception: This function may not converge

    Returns:
        float: The function's root
    """    
    if f(a, *args)*f(b, *args) > 0:
        raise Exception("f(a)f(b) > 0, This function may not converge")

    n = 0
    while n < N_MAX:
        f_a = f(a, *args)
        f_b = f(b, *args)

        c = (a*f_b - b*f_a) / (f_b - f_a)
        f_c = f(c, *args)

        if f_c == 0 or abs(b-a)<epsilon:
            print(SOL_STRING.format(n, abs(b-a), f_c))
            return c

        if f_a * f_c < 0:
            b = c
        else:
            a = c
        
        n += 1
        
    print(NO_SOL_STRING.format(n, f_c))
    return c

@timeit
def modfalsepos(f: typing.Callable[[float], float], a: float, b: float, args: typing.Dict, N_MAX: int = 1000, epsilon: float = 0.001) -> float:
    """Root finding method modified false position

    Args:
        f (typing.Callable[[float], float]): function (callable)
        a (float): left boundary of the root finding interval
        b (float): right boundary of the root finding interval
        args (typing.Dict): function (f) arguments
        N_MAX (int, optional): Maximum number of iterations. Defaults to 1000.
        epsilon (float, optional): Minimum distance between a,b. Defaults to 0.001.

    Raises:
        Exception: This function may not converge

    Returns:
        float: The function's root
    """    
    if f(a, *args) * f(b, *args) > 0:
        raise Exception("f(a)f(b) > 0, This function may not converge")
    
    prev_sign = None  # Variable to keep track of the sign of f_a * f_c
    for n in range(N_MAX + 1):
        f_a = f(a, *args)
        f_b = f(b, *args)
        c = (a * f_b - b * f_a) / (f_b - f_a)
        f_c = f(c, *args)

        if f_c == 0 or abs(b - a) < epsilon:
            print(SOL_STRING.format(n, abs(b-a), f_c))
            return c

        if f_a * f_c < 0:
            if prev_sign == "negative":
                c = (a * f_b - 2 * b * f_a) / (f_b - 2 * f_a)
            prev_sign = "negative"
        else:
            if prev_sign == "positive":
                c = (2 * a * f_b - b * f_a) / (2 * f_b - f_a)
            prev_sign = "positive"

        if prev_sign is None:
            if f_a * f_c > 0:
                prev_sign = "positive" 
            else:
                prev_sign = "negative"

        if f_a * f_c < 0:
            b = c
        else:
            a = c

    return c

@timeit
def rootfindscipy(f: typing.Callable[[float], float], a: float, b: float, args: typing.Dict):
    """Root finding method ridder

    Args:
        f (typing.Callable[[float], float]): function (callable)
        a (float): left boundary of the root finding interval
        b (float): right boundary of the root finding interval
        args (typing.Dict): function (f) arguments
        epsilon (float, optional): Minimum distance between a,b. Defaults to 0.001.

    Raises:
        Exception: This function may not converge

    Returns:
        float: The function's root
    """    
    return ridder(f, a, b,  args, full_output=True)

# MAIN

def main():
    """
    magic numbers
    """
    # set parameter list 
    params = [{"lmbda" : 0.5, "mu" : 1}, {"lmbda" : 1, "mu" : 3}, {"lmbda" : 5, "mu" : 1}, {"lmbda" : 10, "mu" : 3}]

    # plot function with parameters
    plotfunf(params)
    #plt.savefig("figures/pdf_plot.pdf", format="pdf")
    plt.show()

    # bounds of integration
    a = 0
    b = 1000
    print(romberg(funf, a, b, 25, params[0]))
    print(riemann(funf, a, b, 0.001, params[0]))
    print(integratescipy(funf, a, b, params[0]))

    # fetch data from disk and calculate parameter
    arr = np.genfromtxt("xdata.txt")
    mu_hat = arr.mean()

    # create plot object and set bounds and delta
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    leftBound = 0.1
    rightBound = 3
    delta = 0.01
    axes[0] = plotfunl(axes, mu_hat, arr, leftBound, rightBound, delta)
    axes[1] = plotfunderivativel(axes, mu_hat, arr, leftBound, rightBound, delta)

    plt.tight_layout()
    #plt.savefig("figures/log_likelihood.pdf", format="pdf")
    plt.show()

    # Use root finding methods
    print(bisection(funderivativel, 0.1, 3, (mu_hat, arr)))
    print(modfalsepos(funderivativel, 0.1, 3, (mu_hat, arr)))
    print(falsepos(funderivativel, 0.1, 3, (mu_hat, arr)))

if __name__ == '__main__':
    main()