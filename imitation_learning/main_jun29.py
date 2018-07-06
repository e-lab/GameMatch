import os, sys
import subprocess
from multiprocessing import Process, Pool, log_to_stderr
import contextlib
import logging

from experiment import exp2
import random

def test(arg):
    print(arg)


def main():
    sys.stderr = open('process_logging.txt', 'a')
    sys.stdout = open('exp_out.txt', 'a')
    log_to_stderr(logging.DEBUG) 

    m= 'ALSTM'
    params_list = [(m, 16, '0'), \
                    (m, 32, '1'), \
                    (m, 64, '2'), \
                    (m, 128, '3')]

    for param in params_list:
        job = Process(target=exp2, args=param, name=param[-1])
        job.start()

if __name__ == '__main__':
    main()
