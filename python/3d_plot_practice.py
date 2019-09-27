import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def formula(x, y, prime=False):
    if prime:
        return np.array([2 * x, 2 * y])
    else:
        return pow(x, 2) + pow(y, 2)

def DrwaPlot(func, minimum, x, y):
    fig = plt.figure()

    # 3D Axes Instance
    ax = fig.gca(projection='3d')

    C = np.random.randint(0, 5, 3)

    a = np.array([])
    b = np.array([])
    for ele in minimum:
        a = np.append(a, ele[0])
        b = np.append(b, ele[1])

    # a = np.array([1, 0.8, 0.64])
    # b = np.array([-3, -2.4, -1.92])

    ax.plot(a, b, func(a, b), alpha=1.0, marker='.', markersize='3', color='red', linewidth='1')
    ax.scatter(0, 0, func(0, 0), alpha=1.0, s=10, color="blue")
    surf= ax.plot_surface(x, y, func(x, y), alpha=0.5, rstride=5, cstride=5,cmap=cm.binary, antialiased=True)

    ax.set_title(r'$f(x, y)=x^{2}+y^{2}$')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_zlim(-8,8)


    fig.colorbar(surf, shrink=0.5, aspect=10)  # colour bar
    ax.view_init(elev=50, azim=70)               # elevation & angle

    plt.show()

def GradientDescent(x, grad, lr=0.1):
    return x - (lr * grad)

def SearchMinimum(func, init_minimum, step=30, lr=0.1):
    # search minimum value using Gradient Descent
    minimum = init_minimum
    minimum_history = [init_minimum]
    for i in range(step):
        grad = func(minimum[0], minimum[1], True)
        minimum = GradientDescent(minimum, grad, lr)
        minimum_history.append(minimum)
    return minimum_history


x = np.linspace(-3,3,100,True)
y = np.linspace(-3,3,100,True)

# create "base grid"
x, y = np.meshgrid(x, y)

minimum_history = SearchMinimum(formula, np.array([[-3], [-3]]))
print(minimum_history)

DrwaPlot(formula, minimum_history, x, y)
