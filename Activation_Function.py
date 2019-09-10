import os
import matplotlib.pyplot as plt
import numpy as np
import math

def Sigmoid(x):
    # Sigmoid = lambda x: 1 / (1 + np.exp(-x))
    return 1 / (1 + np.exp(-x))

def HyperbolicTangent(x):
    # return 2 / (1 + np.exp(-2 * x)) - 1
    return 2 * Sigmoid(2 * x) -1

def Relu(x):
    return np.maximum(x, 0)

def LeakyRelu(x):
    return np.maximum(x * 0.01, 0)

def PRelu(x):
    alpha = 3
    return np.maximum(x * alpha, x)

def ELU(x):
    return 0

def EachFunctinGraph() :
    return 0

def MergeFunctionGraph(**kwarg):
    plt.figure(figsize=(8, 8))

    for key, value in kwarg.items():
        plt.plot(x, value, label=key, linewidth='1.0', linestyle="-")

    plt.grid(True)
    plt.title('Activation Function')
    plt.legend(loc='lower right')
    '''
    Equation Display
    # plt.text(4, 0.8, eq, fontsize=15)
    '''

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))

    plt.show()

def test(**kwarg):
    for key, value in kwarg.items():
        print(key)
        print(value)

if __name__ == '__main__':
    x = np.linspace(-10, 10, 256, endpoint=True)

    '''
    latex equation
    sigmoid_eq =  r'$\sigma(x)=\frac{1}{1+e^{-x}}$'
    tanh_eq =  r'$\tanh (x)=\frac{2}{1+e^{-2 x}}-1$'
    '''

    sigmoid = Sigmoid(x)
    tanh = HyperbolicTangent(x)

    x = np.linspace(-2, 2, 256, endpoint=True)
    relu = Relu(x)
    x = np.linspace(-20, 20, 256, endpoint=True)
    leaky_relu = LeakyRelu(x)
    x = np.linspace(-1, 1, 256, endpoint=True)
    p_relu = PRelu(x)

    '''
    Overall Activation Function Visualization
    '''
    MergeFunctionGraph(sigmoid=sigmoid, tanh=tanh, relu=relu, leakyRelu=leaky_relu, PRelu=p_relu)
