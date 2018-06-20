import subprocess

def run_bash(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, executable='/bin/bash')
    out = p.stdout.read().strip()
    return out  # This is the stdout from the shell command


def main():
    max_job_num = 3
    run_bash('ts -S %d'%max_job_num)

    models = ['AFC', 'ALSTM', 'ALSTM1']

    for m in models:
        job_cmd = 'python3 experiment.py ' + m

        submit_cmd = "ts bash -c '%s'" % job_cmd
        run_bash(submit_cmd)


if __name__ == '__main__':
    main()
