import math
import random as random
import numpy as np

def identity(x):
    return x

def heaviside(x, alpha:float = 0.0):
    return 1.0 if x >= alpha else 0.0

def binary_step(x):
    return heaviside(x, 0.0)

def symmetric_heaviside(x, alpha):
    return 1.0 if x >= alpha else -1.0 

def sigmoid(x):
    return 1.0 / (1 + math.e ** (x * -1))

def logistic(x):
    return sigmoid(x)

def tanh(x):
    return math.tanh(x)

def sqnl(x):
    ''' Square Non-linearity '''
    return x if x < -2.0 else x + ((x ** 2) / x) if -2.0 <= x < 0 else x - ((x ** 2)/ x) if 0 <= x < 2.0 else 1

def square_non_linearity(x):
    return sqnl(x)

def arctan(x):
    return math.atan(x)

def arsinh(x):
    return math.asinh(x)

def softsign(x):
    return x / (1 + abs(x))

def elliotsig(x):
    return softsign(x)

def inverse_sqrt_unit(x, alpha):
    return x / math.sqrt(1 + alpha * x ** 2)

def isru(x, alpha):
    return inverse_sqrt_unit(x, alpha)

def inverse_sqrt_linear_unit(x, alpha):
    return x if x >= 0 else inverse_sqrt_unit(x, alpha)

def isrlu(x, alpha):
    return inverse_sqrt_linear_unit(x, alpha)

def piecewise_linear_unit(x, alpha, c):
    return max(alpha * (x + c) - c, min(alpha * (x - c) + c), x)

def plu(x, alpha, c):
    return piecewise_linear_unit(x, alpha, c)

def rectified_linear_unit(x):
    return 0 if x<= 0 else x

def relu(x):
    return rectified_linear_unit(x)

def bipolar_rectified_linear_unit(x, i = None):
    if __iter__ in x:
        ''' Iterable solution '''
        assert i is None, f'Evaluating iterable brelu requires index i be None'
        return relu(x) * np.resize([1,-1], len(x))
    else:
        ''' Non-iterable solution '''
        assert i is not None, f'Evaluating non-iterable brelu requires index i'
        return relu(x) if i % 2 == 0 else relu(x) * -1

def brelu(x, i):
    return x,i

def parametric_rectified_linear_unit(x, alpha):
    return x if x >= 0 else x * alpha

def prelu(x, alpha):
    return parametric_rectified_linear_unit(x, alpha)

def leaky_rectified_linear_unit(x):
    return parametric_rectified_linear_unit(x, 0.01)

def leaky_relu(x):
    return leaky_rectified_linear_unit(x)

def random_leaky_relu(x):
    return parametric_rectified_linear_unit(x, random.uniform(0.0, 1.0))

def rrelu(x):
    return random_leaky_relu(x)

def gaussian_error_linear_unit(x, approx = True):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3))) if approx else 0.5 * x * (1 + math.erf(x / math.sqrt(2)))

def gelu(x, approx = True):
    return gaussian_error_linear_unit(x, approx)

def exponential_linear_unt(x, alpha):
    return x if x > 0 else alpha * ((math.e ** x) - 1)

def elu(x, alpha):
    return exponential_linear_unt(x, alpha)

def scaled_exponential_linear_unit(x, alpha, sigma):
    return sigma * exponential_linear_unt(x, alpha)

def selu(x, alpha, sigma):
    return scaled_exponential_linear_unit(x, alpha, sigma)

def softplus(x):
    return math.log(1 + math.e ** x)

def bent_identity(x):
    return (math.sqrt(x ** 2 + 1) - 1) / 2 + x

def sigmoid_linear_unit(x):
    return x / (1 + math.e ** x)

def silu(x):
    return sigmoid_linear_unit(x)

def soft_exponential(x, alpha):
    return x if alpha == 0 else math.e ** (alpha * x) / alpha + alpha if alpha > 0 else math.log(1 - alpha * (x + alpha)) / alpha * -1

def soft_clipping(x, alpha):
    assert alpha != 0, 'Non-zero alpha required'
    return (1 / alpha)  * math.log((1 + math.e ** (alpha * x)) / (1 + math.e ** (alpha * (x - 1))))

def sin(x):
    return math.sin(x)

def sinusoid(x, limited = False):
    return sin(x) if not limited else -1 if x < (math.pi / -2) else 1 if x > (math.pi / 2) else sin(x) 

def sinc(x):
    return 1 if x == 0 else math.sin(x) / x

def gaussian(x):
    return math.e ** (-1 * x ** 2)

def abs(x):
    return abs(x)