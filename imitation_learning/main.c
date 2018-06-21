#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<sys/types.h>
#include<sys/wait.h>

#define NUM_PROCESS 3

int main() {
    char *devices[NUM_PROCESS] = {"1","2","NUM_PROCESS"};
    char *models[NUM_PROCESS] = {"AFC", "ALSTM", "ALSTM1"};

    pid_t pids[NUM_PROCESS];
    int i;

    for (i = 0; i < NUM_PROCESS; i++) {
        char *args[6] = {"experiment.py", "-m", models[i], "-d", devices[i], NULL};

        pids[i] = fork();

        if (!pids[i]) {
            execvp("python3", args);
            exit(1);
        }

    }

    int status;
    pid_t pid;
    int n = NUM_PROCESS;

    while (n > 0) {
        pid = wait(&status);
        printf("Child %ld\tstatus 0x%x\n", (long) pid, status);
        n--;
    }

    return 0;
}
        
