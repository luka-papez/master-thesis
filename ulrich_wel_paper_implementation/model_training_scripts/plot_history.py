import pickle
import os
import numpy as np
import sys
from natsort import natsorted
import matplotlib.pyplot as plt

def load_points(history_folder, model_name, only_iterations, sparsity, **kwargs):
    points = [pickle.load(open(os.path.join(history_folder, point_filename), 'rb'))
        for point_filename in natsorted(os.listdir(history_folder))
            if point_filename.endswith('pickle')
            and point_filename.startswith('history')
            and model_name in point_filename
            and (point_filename.split('-')[-1].split('.')[0] == '1' if only_iterations else True)]
    return points[::sparsity]

if __name__ == '__main__':
    args = { 'sparsity':1, 'history_folder':'../history', 'model_name':'LSTM', 'only_iterations':True, 'plot_bars':False }
    args.update(eval('{' + ','.join([x if ':' in x else '"{}":{}'.format(*(x.split('='))) for x in sys.argv[1:]]) + '}'))
    print(args)

    # (train_loss, train_acc, val_loss, val_acc)
    points = load_points(**args)
    train_acc =  [x[1] * 100 for x in points]
    val_acc   =  [x[3] * 100 for x in points]

    train_loss = [x[0] for x in points]
    val_loss   = [x[2] for x in points]

    index = [int(x + 1) for x in range(len(points))]
    iteration_bars = [68396 / 1024 * v for v in range(1, 21)] if args['plot_bars'] else []

    plt.plot(index, train_acc, 'b', index, val_acc, 'g')
    plt.ylabel('%')
    plt.legend(['train_accuracy', 'validation_accuracy'])
    for iter_bar in iteration_bars:
        plt.axvline(x=iter_bar, color='gray', linewidth=0.5)
    plt.show()

    plt.plot(index, train_loss, 'b--', index, val_loss, 'g--')
    plt.legend(['train_loss', 'validation_loss'])
    for iter_bar in iteration_bars:
        plt.axvline(x=iter_bar, color='gray', linewidth=0.5)
    plt.show()
