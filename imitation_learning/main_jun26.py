import os, sys
import subprocess
from multiprocessing import Process, Pool, log_to_stderr
import contextlib
import logging

from experiment import exp2
import random

def main():
    num_pool_workers= 4

    random.seed()

    sys.stderr = open('process_logging.txt', 'a')
    log_to_stderr(logging.DEBUG) 

    m= 'ALSTM'
    params_list = [(m, 16, '0'), \
                    (m, 32, '1'), \
                    (m, 64, '2'), \
                    (m, 128, '3')]

    with contextlib.closing(Pool(num_pool_workers)) as po: # This ensures that the processes get closed once they are done
        pool_results = po.map_async(exp2, \
                            ((param1, param2, param3) \
                            for param1, param2, param3 in params_list))
        results_list = pool_results.get()


if __name__ == '__main__':
    main()
