import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath('../')

class Evaluation():
    def __init__(self):
        self._results = np.array([])
        self._prec = []
        self._rec = []

    def CreateSampleResult(self):
        data = pd.read_csv(os.path.join(ROOT_DIR, 'files/evaluation_metric_sample.csv'))
        self._results = np.array(data)

    def ComputePrecisionAndRecall(self):
        acc_tp = 0
        acc_fp = 0
        num_of_object = len(self._results)
        results = self._results

        while len(results) > 0:
            idx = np.argmax(results[:, 2])
            if results[idx, 3] == 'TP':
                acc_tp +=1
            elif results[idx, 3] == 'FP':
                acc_fp +=1

            results = np.delete(results, (idx), axis=0)
            self._prec.append(acc_tp / (acc_tp + acc_fp))
            self._rec.append(acc_tp / num_of_object)

    def GetPrecision(self):
        return self._prec

    def GetRecall(self):
        return self._rec

    def VisualizePlot(self, title='Default', ax=None):

        is_auto_show = False
        if ax is None:
            fig, ax = plt.subplots(1)
            fig.canvas.set_window_title('dd')
            is_auto_show= True

        ax.plot(self._rec, self._prec, linewidth='1.0', linestyle="-")
        ax.grid(alpha=.4,linestyle='--')
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title('Precision & Recall Graph')

        if is_auto_show:
            plt.show()

    def ComputAveragePrecision():
        print('compute Averate Precision')

    def ComputeElevenPointInterpolation():
        print('eleven point interpolation')

    def ComputeEveryPointInterpolation():
        print('every point interpolation')


if __name__ == '__main__':
    evaluation = Evaluation()
    evaluation.CreateSampleResult()
    evaluation.ComputePrecisionAndRecall()

    print(' - Precision\n', evaluation.GetPrecision())
    print('- Recall\n', evaluation.GetPrecision())

    fig, ax = plt.subplots(1, 3, figsize=(10,5))
    fig.canvas.set_window_title('Precision & Recall Graph')
    evaluation.VisualizePlot(ax=ax[0])

    plt.show()
