import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# def formula(x, y, prime=False):
#     if prime:
#         return np.array([2 * x, 2 * y])
#     else:
#         return pow(x, 2) + pow(y, 2)

def formula(x, y, prime=False):
    if prime:
        return np.array([x / 5.0, 2.0*y])
    else:
        return x**2 / 10.0 + y**2

def DrwaPlot(func, params, x, y):
    fig = plt.figure(figsize=(12, 12))
    # 3D Axes Instance
    ax = fig.gca(projection='3d')

    for key, val in params.items():
        ax.plot(val[0], val[1], func(val[0], val[1]), label=key, alpha=1.0, marker='.', markersize='5', linewidth='0.5')

    ax.scatter(0, 0, func(0, 0), label='Minimum', alpha=1.0, s=10, color="blue")
    surf= ax.plot_surface(x, y, func(x, y), alpha=0.5, rstride=5, cstride=5,cmap=cm.binary, antialiased=True)
    ax.set_title(r'$f(x)=\frac{1}{10} x^{2}+y^{2}$')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # fig.colorbar(surf, shrink=0.5, aspect=10)  # colour bar
    ax.view_init(elev=50, azim=70)               # elevation & angle
    ax.legend(loc='upper left')

    plt.show()

def GradientDescent(param, grad, cache=np.zeros((2, 1), dtype='float64'), lr=0.01):
    param -= lr * grad

    return param, cache

def Momentum(param, grad, cache=np.zeros((2, 1), dtype='float64'), lr=0.01, m=0.9):
    cache = (m * cache) - (lr * grad);
    param += cache

    return param, cache

def AdaGrad(param, grad, cache=np.zeros((2, 1), dtype='float64'), lr=0.01):
    cache += pow(grad, 2)
    param += - (lr * grad) / (np.sqrt(cache) + 1e-8)

    return param, cache


def Adam(param, grad, cache=np.zeros((2, 1), dtype='float64'), lr=0.01, beta1=0.9, beta2=0.999):
    m = beta1 * m + (1-beta1) * param
    v = beta2 * v + (1-beta2) * (pow(param,2))
    param += -lr * m / (np.sqrt(v) + 1e-8)

    return param, cache


def SearchParam(func, init_param=np.array([[-3.0], [2.0]]), step=20, lr=0.1):
    # search minimum value using Gradient Descent
    params = init_param
    params_history = init_param
    cache = np.zeros((2, 1), dtype='float64')
    for i in range(step):
        grad = formula(params[0].copy(), params[1].copy(), True)
        params, cache = func(params.copy(), grad.copy(), cache=cache, lr=lr)
        params_history = np.hstack((params_history, params))

    return params_history

x = np.linspace(-3.0, 3.0, 100,True)
y = np.linspace(-3.0, 3.0, 100,True)

# create "base grid"
x, y = np.meshgrid(x, y)

optimizer = dict()

optimizer['GD'] = SearchParam(GradientDescent, lr=0.8)
optimizer['Momentum'] = SearchParam(Momentum, lr=0.1)
optimizer['AdaGrad'] = SearchParam(AdaGrad, lr=1.0)
# optimizer['Adam'] = SearchParam(Adam, lr=1.0)

DrwaPlot(formula, optimizer, x, y)
