//#include <iostream>
//#include <thread>
//#include <time.h>
//#include  <stdio.h>
//
///*--- xミリ秒経過するのを待つ ---*/
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>

#define N_THREADS 1

int g;
int x;

int sleep_(unsigned long x)
{
    clock_t  s = clock();
    clock_t  c;
    do {
        if ((c = clock()) == (clock_t)-1)       /* エラー */
            return (0);
    } while (1000UL * (c - s) / CLOCKS_PER_SEC <= x);
    return (1);
}
typedef struct thread_arg {
    pthread_t tid;
    int num;
} * thread_arg_t;

void * thread_func(void * _arg) {
    for (int i=0; i<2;i++){
        sleep_(500);
        printf("subthread sleeping..");
        if (i>5) {
            x++;
        }
    }
    printf("subthread end..");
    return 0;
}

double cur_time() {
    struct timeval tp[1];
    gettimeofday(tp, NULL);
    return tp->tv_sec + tp->tv_usec * 1.0E-6;
}

int main()
{
    struct thread_arg args[N_THREADS];
    double t0 = cur_time();
    int i;
    int y;
    /* スレッドを N_THREADS 個作る */
    for (i = 0; i < N_THREADS; i++) {
        args[i].num=y;
        pthread_create(&args[i].tid, NULL,
                       thread_func, (void *)&args[i]);
    }
    /* 終了待ち */
    while (1) {
        for (int i=0; i<15;i++){
            sleep_(1000);
            printf("mainthread sleeping..");
            if (x>0) {
                printf("detect change in x:%d",x);
            }
        }
        break;
    }
    //    for (i = 0; i < N_THREADS; i++) {
    //        pthread_join(args[i].tid, NULL);
    //    }
    double t1 = cur_time();
    printf("OK: elapsed time: %f\n", t1 - t0);
    return 0;
}