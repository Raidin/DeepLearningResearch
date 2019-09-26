import numpy as np
import matplotlib.pyplot as plt

# def func(x, prime=False):
#     if prime:
#         return 2*x + 2
#     else:
#         return pow(x, 2) + 2 * x

def func(x, prime=False):
    if prime:
        return 2*x + 1
    else:
        return  pow(x, 2) + x + 2

# Learning Rate에 따라서 Local Minimum 찾는 속도 체크하기.
def gradient_descent(x, lr=0.8, step_num=100):
    x_history = []

    for i in range(step_num):
        x_history.append(x)

        grad = func(x, True)
        x -= lr * grad

        if grad == 0: break

    return x, x_history

# tangent : 접선
def tangent_line(x):
    d = func(x, True)
    y = func(x) - d * x
    return lambda t: d * t + y

fig = plt.figure(figsize=(10, 10))

x = np.linspace(-4,4,100,True)
plt.plot(x, func(x), label='function', linewidth='1.0', linestyle='-')


local_minimum, local_minimum_history = gradient_descent(-2.0)
print('Minimum Value :', local_minimum)
print('MiniMum History :', local_minimum_history)

# local minimum 그래프 그릴때 해당 지점 점으로 표시해주는 거 넣기..
for i, ele in enumerate(local_minimum_history[:5]) :
    tf = tangent_line(ele)
    plt.plot(x, tf(x), label='{}-th local minimum({:.4f}) tangent line'.format(i, ele), linewidth='1.0', linestyle='--')

plt.legend(loc='upper left')
plt.show()
