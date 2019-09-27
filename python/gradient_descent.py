import numpy as np
import matplotlib.pyplot as plt

# 3차 방정식 예제
def CubicFunc(x, prime=False):
    if prime:
        return 3 * pow(x, 2) + 6 * x - 6
    else:
        return  pow(x, 3) + 3 * pow(x, 2) - 6 * x - 8

def GradientDescent(x, grad, lr=0.1):
    x -= lr * grad
    return x

def SearchMinimum(func, init_minimum, step=10, lr=0.1):
    # search minimum value using Gradient Descent
    minimum = init_minimum
    minimum_history = [init_minimum]
    for i in range(step):
        grad = func(minimum, True)
        minimum = GradientDescent(minimum, grad, lr)
        minimum_history.append(minimum)
    return minimum_history

# 접선 함수
def TangentLine(func, x):
    d = func(x, True)
    y = func(x) - d * x
    return lambda t: d * t + y

def DrawPlot(x, func, minimum, title, is_minimum_list=True):
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot()

    # Function Graph
    plt.plot(x, func(x), c='#000011', label='function', linewidth='1.0', linestyle='-')

    if is_minimum_list:
        for i, ele in enumerate(minimum):
            tf = TangentLine(func, ele)
            plt.plot(x, tf(x), label='{}-th minimum({:.4f})'.format(i, ele), linewidth='1.0', linestyle='--')
            plt.scatter(ele, func(ele), linewidths='1')
    else:
        tf = TangentLine(func, minimum)
        plt.plot(x, tf(x), label='minimum({:.4f})'.format(minimum), linewidth='1.0', linestyle='--')
        plt.scatter(minimum, func(minimum), linewidths='1')

    # Label Box Position
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title(title)
    plt.grid(alpha=.4,linestyle='--')
    plt.ylim(-15, 15)

    # axis 가운데로 정렬
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    plt.show()

def main():
    x = np.linspace(-5, 5, 100,True)
    title = r'$f(x)=x^{3}+3 x^{2}-6 x-8$'

    # 입력된 함수의 최소 값 접선(기울기) 예시
    DrawPlot(x, CubicFunc, 0.7321, title, False)

    # 시작시점을 -1.0으로 했을 때, LearningRate 별 수렴 과정 비교
    minimum_history = SearchMinimum(CubicFunc, -1.0, step=30, lr=0.2)
    re_title = ''
    re_title += title + ' (start :: {}, step :: {}, lr :: {})'.format(-1.0, 30, 0.2)
    DrawPlot(x, CubicFunc, minimum_history, re_title)

    minimum_history = SearchMinimum(CubicFunc, -1.0, step=10, lr=0.1)
    re_title = ''
    re_title += title + ' (start :: {}, step :: {}, lr :: {})'.format(-1.0, 10, 0.1)
    DrawPlot(x, CubicFunc, minimum_history, re_title)

    minimum_history = SearchMinimum(CubicFunc, -1.0, step=10, lr=0.01)
    re_title = ''
    re_title += title + ' (start :: {}, step :: {}, lr :: {})'.format(-1.0, 10, 0.01)
    DrawPlot(x, CubicFunc, minimum_history, re_title)

    minimum_history = SearchMinimum(CubicFunc, -1.0, step=10, lr=0.001)
    re_title = ''
    re_title += title + ' (start :: {}, step :: {}, lr :: {})'.format(-1.0, 10, 0.001)
    DrawPlot(x, CubicFunc, minimum_history, re_title)

    minimum_history = SearchMinimum(CubicFunc, -3.0, step=4, lr=0.01)
    re_title = ''
    re_title += title + ' (start :: {}, step :: {}, lr :: {})'.format(-3.0, 4, 0.01)
    DrawPlot(x, CubicFunc, minimum_history, re_title)

if __name__ == '__main__':
    main()
