import numpy as np
import matplotlib.pyplot as plt
from activation_function import *

class Layer:
    def __init__(self, lr=0.5):
        self.lr = lr

def MeanSquredError(y, t, prime=False):
    if prime:
        return y - t
    else:
        return np.sum(pow(t - y,2))/2

def DrawLossPlot(**loss):
    for key, value in loss.items():
        plt.plot(value, label=key, linewidth='1.0', linestyle="-")
    plt.grid(alpha=.9,linestyle='--')
    plt.legend(loc='upper right', fontsize='large')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.show()


def Train(x, t, params, act_func):
    epoch = 10
    lr = 0.5
    loss = []

    for i in range(epoch):
        # Forward
        ######################################################
        y1 = np.dot(x, params['w1'])
        a1 = act_func(y1)

        y2 = np.dot(a1, params['w2'])
        a2 = act_func(y2)

        out = a2
        loss.append(MeanSquredError(out, t))
        ######################################################

        # Backward
        ######################################################
        dout = MeanSquredError(out, t, True) # dt/da2
        dout = dout * act_func(y2, True) # dt/dy2

        dw2 = np.dot(a1.T, dout)

        dout = np.dot(dout, params['w2'].T)
        dout = dout * act_func(y1, True)

        dw1 = np.dot(x, dout)
        ######################################################

        params['w2'] -= lr * dw2
        params['w1'] -= lr * dw1

        # print('[{}]-iter loss :: {}'.format(i+1, loss[-1]))

    print('\t[TARGET] ::', t)
    print('\t[OUTPUT] ::', out)
    print('\t[LOSS] ::', loss[-1])
    # print('==================== W1 ====================\n', params['w1'])
    # print('==================== W2 ====================\n', params['w2'])

    return loss

def InitializeWeight():
    params = dict()
    params['w1'] = np.array([[0.15], [0.2]], dtype='float16').reshape(1, 2)
    # print('- layer-1_weight(shape:{})\n{}'.format(params['w1'].shape, params['w1']))
    params['w2'] = np.array([[0.40, 0.50],
                   [0.45, 0.55]], dtype='float16')
    # print('- layer-2_weight(shape:{})\n{}'.format(params['w2'].shape, params['w2']))
    return params

def main():
    t = np.array([[0.2, 0.7]], dtype='float16')
    print('- Target ::', t.shape, t)
    x = np.array([[0.7]], dtype='float16')
    print('- Input(shape:{})\n{}'.format(x.shape, x))

    loss_dict = dict()

    # Using Sigmoid Activation
    params = InitializeWeight()
    print(' ============= USING SIGMOID ACTIVATION FUNCTION ============= ')
    loss = Train(x, t, params, Sigmoid)
    loss_dict['Sigmoid'] = loss

    # Using Relu Activation
    params = InitializeWeight()
    print(' ============= USING Tanh ACTIVATION FUNCTION ============= ')
    loss = Train(x, t, params, HyperbolicTangent)
    loss_dict['Tanh'] = loss

    # Using Relu Activation
    params = InitializeWeight()
    print(' ============= USING RELU ACTIVATION FUNCTION ============= ')
    loss = Train(x, t, params, Relu)
    loss_dict['Relu'] = loss

    # Using LeakRelu Activation
    params = InitializeWeight()
    print(' ============= USING LeakyRelu ACTIVATION FUNCTION ============= ')
    loss = Train(x, t, params, LeakyRelu)
    loss_dict['LeakyRelu'] = loss

    # Using LeakRelu Activation
    params = InitializeWeight()
    print(' ============= USING ELU ACTIVATION FUNCTION ============= ')
    loss = Train(x, t, params, ELU)
    loss_dict['ELU'] = loss

    DrawLossPlot(**loss_dict)

if __name__ == '__main__':
    main()
