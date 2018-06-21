import os
import subprocess
from multiprocessing import Process

from experiment import exp

def run_bash(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, executable='/bin/bash')
    out = p.stdout.read().strip()
    return out  # This is the stdout from the shell command


def main():
    max_job_num = 3

    '''
    run_bash('ts -S %d'%max_job_num)
    '''
    # devices = [str(x) for x in range(max_job_num)]
    process = []
    models = ['AFC', 'ALSTM', 'ALSTM1']

    for i, m in enumerate(models):
        '''
        job_cmd = 'python3 experiment.py -m ' + m + ' -d ' + str(i) + ' &'

        # submit_cmd = "ts bash -c '%s'" % job_cmd
        run_bash(job_cmd)
        '''
        process.append(Process(target=exp, args=(m, str(i)), name='p'+str(i)))
        process[i].start()


if __name__ == '__main__':
    main()
