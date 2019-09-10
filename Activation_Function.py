import matplotlib.pyplot as plt
import numpy as np

def Sigmoid(x):
    # Sigmoid = lambda x: 1 / (1 + np.exp(-x))
    return 1 / (1 + np.exp(-x))

def HyperbolicTangent(x):
    # return 2 / (1 + np.exp(-2 * x)) - 1
    return 2 * Sigmoid(2 * x) -1

def Relu(x):
    return np.maximum(x, 0)

def LeakyRelu(x):
    return np.maximum(x * 0.01, x)

def PRelu(x):
    alpha = 2
    return np.maximum(x * alpha, x)

def ELU(x):
    alpha = 2
    return np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x) - 1))

def EachFunctinGraph(*args) :
    plt.figure(figsize=(8, 8))

    plt.plot(x, args[1], label=args[0], linewidth='1.0', linestyle="-")

    plt.grid(True)
    plt.title(args[0])
    plt.legend(loc='upper left')

    '''
    Equation Display
    '''
    plt.text(0, 0, args[2], fontsize=15)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))

    plt.show()

def MergeFunctionGraph(**kwarg):
    plt.figure(figsize=(8, 8))

    for key, value in kwarg.items():
        plt.plot(x, value, label=key, linewidth='1.0', linestyle="-")

    plt.grid(True)
    plt.title('Activation Function')
    plt.legend(loc='upper left')

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))

    plt.show()

if __name__ == '__main__':
    x = np.linspace(-10, 10, 256, endpoint=True)

    '''
    latex equation
    '''
    sigmoid_eq =  r'$\sigma(x)=\frac{1}{1+e^{-x}}$'
    tanh_eq = r'$\tanh (x)=\frac{2}{1+e^{-2 x}}-1$'
    relu_eq = r'$f(x)=\max (0, x)$'
    leaky_relu_ep = r'$f(x)=\max (0.01 x, x)$'
    prelu_eq = r'$f(x)=\max (\alpha x, x)$'
    elu_eq = r'$f(x)=\max (0, x)+\min \left(0, \alpha *\left(e^{\frac{x}{\alpha}}-1\right)\right)$'

    sigmoid = Sigmoid(x)
    EachFunctinGraph('sigmoid', sigmoid, sigmoid_eq);

    tanh = HyperbolicTangent(x)
    EachFunctinGraph('tanh', tanh, sigmoid_eq);

    x = np.linspace(-2, 2, 256, endpoint=True)
    relu = Relu(x)
    EachFunctinGraph('relu', relu, sigmoid_eq);

    leaky_relu = LeakyRelu(x)
    EachFunctinGraph('LeakyRelu', leaky_relu, sigmoid_eq);

    x = np.linspace(-1, 1, 256, endpoint=True)
    p_relu = PRelu(x)
    EachFunctinGraph('PRelu', p_relu, sigmoid_eq);

    elu = ELU(x)
    EachFunctinGraph('ELU', elu, sigmoid_eq);

    '''
    Overall Activation Function Visualization
    '''
    MergeFunctionGraph(sigmoid=sigmoid, tanh=tanh, relu=relu, leakyRelu=leaky_relu, PRelu=p_relu, elu=elu)
