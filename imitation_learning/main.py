import os
import subprocess
from multiprocessing import Process

from experiment import exp

def main():
    max_job_num = 3

    process = []
    models = ['AFC', 'ALSTM']

    for i, m in enumerate(models):
        process.append(Process(target=exp, args=(m, str(i)), name='p'+str(i)))
        process[i].start()


if __name__ == '__main__':
    main()
