# plot statistics from the experiments
# Author: Ruihang Du
# email: du113@purdue.edu

import csv
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

FILES = glob('/home/du113/summer_18/stats/2rooms/*july19*')

def plot():
    for fname in FILES:
        with open(fname, 'r') as file:
            lines = file.readlines()

        datalist = []
        buffer = []

        for line in lines:
            if line == '******\n':
                if buffer:
                    datalist.append(buffer)
                    buffer = []
            else:
                buffer.append(float(line.split(',')[1].strip().lstrip('tensor(')))

        datalist = np.array(datalist)

        stat = np.mean(datalist, axis=0)
        x = np.arange(0, 105, 5)

        plt.plot(x, stat, label=fname.split('seq')[1].rstrip('.csv'))
    plt.xlabel('epochs')
    plt.ylabel('inference accuracy')
    plt.title('AlexNet + BN + LSTM + FC in 2 room')
    plt.legend(loc='upper left')
    # plt.title('AACBase + BN + LSTM in 2 rooms')
    plt.show()

    # plt.savefig('../plots/alexbnlstmfc_2room.png')

    return
    df = list(csv.reader(file, delimiter=','))
    x = [int(d[0]) for d in df]
    y = [float(d[1].lstrip('tensor(')) for d in df]

    print(x)
    print(y)


if __name__ == '__main__':
    plot()
