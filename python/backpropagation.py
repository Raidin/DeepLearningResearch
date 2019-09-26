import numpy as np
import matplotlib.pyplot as plt
import copy
from activation_function import *

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

def Forward(x, weigths, act):
    result = []

    for i, w in enumerate(weigths):
        ele = dict()
        ele['x'] = x
        ele['w'] = w
        ele['y'] = np.dot(x, w)
        ele['a'] = act(ele['y'])
        x = ele['a']
        result.append(ele)

    return result

def Backward(forward, t, act):
    dout = MeanSquredError(forward[-1]['a'], t, True) # dt/da2
    forward.reverse()
    result = []

    for f in forward:
        dout = dout * act(f['y'], True) # dt/dy2
        result.append(np.dot(f['x'].T, dout))
        dout = np.dot(dout, f['w'].T)

    return result

def GradientDescent(weigths, grad, lr=0.5):
    grad.reverse()

    for i, w in enumerate(weigths):
        w -= lr * grad[i]

def Train(x, t, params, epoch=200, act_func=Sigmoid, is_visible_loss=False):
    loss = []

    for i in range(epoch):
        # Forward
        forward = Forward(x, params, act_func)
        out = forward[-1]['a']
        loss.append(MeanSquredError(out, t))

        # Backward
        grad = Backward(forward, t, act_func)

        # Gradient Descent
        GradientDescent(params, grad)

        if is_visible_loss : print('[{}]-iter loss :: {}'.format(i+1, loss[-1]))

    print('\t[TARGET] ::', t)
    print('\t[OUTPUT] ::', out)
    print('\t[LOSS] ::', loss[-1])
    return loss

def InitializeWeight(is_fixed=True, in_size=1, hidden_size=2, out_size=2, layer_size=2):
    weights = []
    weight_init_std = 0.1

    if is_fixed:
        weights.append(np.array([[0.15, 0.2]], dtype='float16'))
        weights.append(np.array([[0.40, 0.50],[0.45, 0.55]], dtype='float16'))
    else:
        np.random.seed(777)
        for i in range(layer_size):
            weights.append(weight_init_std * np.random.randn(in_size, hidden_size))
            in_size = hidden_size
            hidden_size = out_size if i == layer_size-1 else hidden_size

    return weights

def main():
    t = np.array([0.2, 0.7], dtype='float16')
    x = np.array([[0.7]], dtype='float16')
    print('- Input ::', x.shape)
    weight_init = InitializeWeight(in_size=x.shape[1], out_size=t.ndim)
    print('- Weight\n', weight_init)

    return 0

    loss_dict = dict()
    activation_func = [Sigmoid, HyperbolicTangent, Relu, LeakyRelu, ELU]

    # Activation Function을 다르게 적용 할지 유무 결정
    is_apply_each_act = False

    if is_apply_each_act:
        # Apply Each Activation Function
        for act in activation_func:
            print(' ============= USING {} ACTIVATION FUNCTION ============= '.format(act.__name__))
            loss = Train(x, t, copy.deepcopy(weight_init), act_func=act)
            loss_dict[act.__name__] = loss
    else:
        act = Sigmoid
        print(' ============= USING {} ACTIVATION FUNCTION ============= '.format(act.__name__))
        loss = Train(x, t, copy.deepcopy(weight_init), act_func=act)
        loss_dict[act.__name__] = loss

    DrawLossPlot(**loss_dict)

if __name__ == '__main__':
    main()
