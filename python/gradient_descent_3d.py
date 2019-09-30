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
        return np.array([x / 10.0, 2.0*y])
    else:
        return x**2 / 20.0 + y**2

def DrwaPlot(func, params, x, y):
    fig = plt.figure(figsize=(12, 12))
    # 3D Axes Instance
    ax = fig.gca(projection='3d')

    for key, val in params.items():
        ax.plot(val[0], val[1], func(val[0], val[1]), label=key, alpha=1.0, marker='.', markersize='5', linewidth='0.5')

    ax.scatter(0, 0, func(0, 0), label='Minimum', alpha=1.0, s=10, color="blue")
    surf= ax.plot_surface(x, y, func(x, y), alpha=0.5, rstride=5, cstride=5,cmap=cm.binary, antialiased=True)
    ax.set_title(r'$f(x)=\frac{1}{20} x^{2}+y^{2}$')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # fig.colorbar(surf, shrink=0.5, aspect=10)  # colour bar
    ax.view_init(elev=50, azim=70)               # elevation & angle
    ax.legend(loc='upper left')

    plt.show()

def GradientDescent(x, grad, factor=np.zeros((2, 1), dtype='float64'), lr=0.01):
    return x - (lr * grad), factor

def Momentum(x, grad, factor=np.zeros((2, 1), dtype='float64'), lr=0.01, m=0.9):
    factor = (m * factor) - (lr * grad);
    x = x + factor
    return x, factor

def AdaGrad(x, grad, factor=np.zeros((2, 1), dtype='float64'), lr=0.01):
    factor = factor + (grad * grad)
    x = x - (lr * grad) / (np.sqrt(factor) + 1e-7)
    return x, factor

def SearchParam(func, init_param=np.array([[-7.0], [2.0]]), step=20, lr=0.1):
    # search minimum value using Gradient Descent
    minimum = init_minimum
    minimum_history = init_minimum

    factor = np.zeros((2,1), dtype='float64')
    for i in range(step):
        grad = formula(minimum[0], minimum[1], True)
        minimum, factor = func(minimum, grad, factor=factor, lr=lr)
        minimum_history = np.hstack((minimum_history, minimum))

    return minimum_history

x = np.linspace(-7,7,100,True)
y = np.linspace(-3,3,100,True)

# create "base grid"
x, y = np.meshgrid(x, y)

params = dict()

params['SGD'] = SearchParam(GradientDescent, lr=0.95)
params['Momentum'] = SearchParam(Momentum, lr=0.1)
params['AdaGrad'] = SearchParam(AdaGrad, lr=1.5)

DrwaPlot(formula, params, x, y)

