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
        self._eleven_prec = []
        self._eleven_rec = []
        self._every_prec = []
        self._every_rec = []
        self._AP = 0
        self.num_of_object = 0

    def GetPrecision(self):
        return self._prec

    def GetRecall(self):
        return self._rec

    def GetAP(self):
        return self._AP

    def CreateSampleResult(self):
        data_path = os.path.join(ROOT_DIR, 'files/evaluation_metric_sample_3.csv')
        data = pd.read_csv(data_path)
        print(pd.DataFrame(data))

        '''
        * Num of object
          - sample 1 : 19
          - sample 2 : 15
          - sample 2 : 5
        '''
        num_of_objects = [19, 15, 5]
        self.num_of_object = num_of_objects[int(os.path.splitext(os.path.basename(data_path))[0].split('_')[-1]) - 1]
        self._results = np.array(data)

    def ComputePrecisionAndRecall(self):
        acc_tp = 0
        acc_fp = 0
        results = self._results

        while len(results) > 0:
            idx = np.argmax(results[:, 2])
            if results[idx, 3] == 'TP':
                acc_tp +=1
            elif results[idx, 3] == 'FP':
                acc_fp +=1

            results = np.delete(results, (idx), axis=0)
            self._prec.append(acc_tp / (acc_tp + acc_fp))
            self._rec.append(acc_tp / self.num_of_object)

    def VisualizePlot(self, mode='default', title='Standard', ax=None):

        is_auto_show = False
        if ax is None:
            fig, ax = plt.subplots(1)
            fig.canvas.set_window_title('Precision & Recall Graph')
            is_auto_show= True

        x = self._rec
        y = self._prec
        drawstyle = 'default'

        if mode is 'eleven':
            x = self._eleven_rec
            y = self._eleven_prec
            drawstyle = 'steps-post'
        elif mode is 'every':
            x = self._every_rec
            y = self._every_prec
            drawstyle = 'steps-post'

        ax.plot(x, y, linewidth='1.0', linestyle="-", drawstyle=drawstyle)
        ax.grid(alpha=.4,linestyle='--')
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)

        if is_auto_show:
            plt.show()

    def ComputeElevenPointInterpolation(self):
        self._eleven_rec = np.arange(0, 1.1, 0.1)
        group = np.stack([evaluation.GetPrecision(), evaluation.GetRecall()], axis=1)
        for i in self._eleven_rec:
            samples = np.where(group[:, 1] >= i, group[:,0], 0)
            self._eleven_prec.append(np.max(samples))

        self._AP = np.mean(self._eleven_prec)
        print('11-point interpolation AP :: {:0.2f}%'.format(self._AP * 100))

    def ComputeEveryPointInterpolation(self):
        self._every_rec = np.unique(self._rec)
        group = np.stack([self._prec, self._rec], axis=1)
        for i in self._every_rec:
            samples = np.where(group[:, 1] >= i, group[:,0], 0)
            self._every_prec.append(np.max(samples))

        # Compute AP
        self._every_rec = np.insert(self._every_rec, 0, 0.0)
        self._AP = 0
        for i in range(1, len(self._every_rec)):
            self._AP += (self._every_rec[i] - self._every_rec[i-1]) * self._every_prec[i-1]
        self._every_rec = np.delete(self._every_rec, 0)
        # Area under curve(AUC)
        print('every-point interpolation AP :: {:0.2f}%'.format(self._AP * 100))


if __name__ == '__main__':
    # Init Evaluation
    evaluation = Evaluation()
    # Create Sample data
    evaluation.CreateSampleResult()
    # Copmute Default precisin and recall
    evaluation.ComputePrecisionAndRecall()

    # print(' - Precision\n', evaluation.GetPrecision())
    # print(' - Recall\n', evaluation.GetRecall())

    fig, ax = plt.subplots(3, 1, figsize=(5, 10))
    fig.canvas.set_window_title('Precision & Recall Graph')
    # Display default precision and recall graph
    evaluation.VisualizePlot(ax=ax[0])

    # Compute 11-point interpolation
    evaluation.ComputeElevenPointInterpolation()
    # Display 11-point interpolation precision and recall graph
    title = '11-point interpolation(AP::{:0.2f}%)'.format(evaluation.GetAP() * 100)
    evaluation.VisualizePlot(ax=ax[1], mode='eleven', title=title)

    # Compute every-point interpolation
    evaluation.ComputeEveryPointInterpolation()
    title = 'Every-point interpolation(AP::{:0.2f}%)'.format(evaluation.GetAP() * 100)
    evaluation.VisualizePlot(ax=ax[2], mode='every', title=title)

    plt.tight_layout()
    plt.show()
