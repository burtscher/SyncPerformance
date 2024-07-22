/*
This file is part of SyncPerformance, a testing framework that measures the execution time of single synchronization primitives in CUDA and OpenMP.

BSD 3-Clause License

Copyright (c) 2024, Brandon Alexander Burtchell and Martin Burtscher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/SyncPerformance.

Publication: This work is described in detail in the following paper.
Brandon Alexander Burtchell and Martin Burtscher. "Characterizing CUDA and OpenMP Synchronization Primitives." Proceedings of the IEEE International Symposium on Workload Characterization. September 2024.
*/


#include <cstdio>
#include <sys/time.h>
#include <limits>

#include "omp_macros.h"
#include "math_helpers.h"
#include "config.h"

template <typename T>
static void func1(const int n_threads, const int n_iter, double* const timer);

template <typename T>
static void func2(const int n_threads, const int n_iter, double* const timer);

template <typename T>
static void run_test(const int n_threads, const int n_iter) {
  double max1 [N_RUNS], max2 [N_RUNS];
  double min1 [N_RUNS], min2 [N_RUNS];
  double med1 [N_RUNS], med2 [N_RUNS];
  double avg1 [N_RUNS], avg2 [N_RUNS];

  const int n_timers = n_threads;
  double timer1 [n_timers];
  double timer2 [n_timers];

  for (int run = 0; run < N_RUNS; run++) {
    int attempt;
    for (attempt = 0; attempt < N_ATTEMPTS; attempt++) {
      func1<T>(n_threads, n_iter, timer1);
      max1[run] = *std::max_element(timer1, timer1 + (n_timers));
      min1[run] = *std::min_element(timer1, timer1 + (n_timers));
      med1[run] = median(timer1, n_timers);
      avg1[run] = arithmetic_mean(timer1, n_timers);

      func2<T>(n_threads, n_iter, timer2);
      max2[run] = *std::max_element(timer2, timer2 + (n_timers));
      min2[run] = *std::min_element(timer2, timer2 + (n_timers));
      med2[run] = median(timer2, n_timers);
      avg2[run] = arithmetic_mean(timer2, n_timers);

      if (max2[run] > max1[run]) {
        break; // attempt succeeded
      }
    }
#ifdef FINE_GRAIN_RESULTS_MODE
    if (run == 1) {  // only get stats on 2nd run
      record_all_thread_runtimes_omp(timer1, timer2, n_timers, n_threads, n_iter);
    }
#endif
    if (attempt >= N_ATTEMPTS) {
      printf("WARNING: never got a positive reading for run %d in %d attempts, test may be faulty", run + 1, N_ATTEMPTS);
    }
  }

  const double runtime_all_iters = median(max2, N_RUNS) - median(max1, N_RUNS);
  const double max_per_op = runtime_all_iters / (double)n_iter / (double)N_UNROLL / (double)N_OPS_PER_TEST_UNIT;
  const double min_per_op = (median(min2, N_RUNS) - median(min1, N_RUNS)) / (double)n_iter / (double)N_UNROLL / (double)N_OPS_PER_TEST_UNIT;
  const double med_per_op = (median(med2, N_RUNS) - median(med1, N_RUNS)) / (double)n_iter / (double)N_UNROLL / (double)N_OPS_PER_TEST_UNIT;
  const double avg_per_op = (median(avg2, N_RUNS) - median(avg1, N_RUNS)) / (double)n_iter / (double)N_UNROLL / (double)N_OPS_PER_TEST_UNIT;

  printf("runtime of all iter: %0.16lf\n", runtime_all_iters);
  printf("max per op: %0.16lf\n", max_per_op);
  printf("min per op: %0.16lf\n", min_per_op);
  printf("med per op: %0.16lf\n", med_per_op);
  printf("avg per op: %0.16lf\n", avg_per_op);
}

void process_args(int argc, char* argv[], int &type_mode, int &n_threads, int &n_iter) {
  if (argc < 4) {printf("USAGE: %s type_mode n_threads n_iter\n", argv[0]); exit(-1);}

  type_mode = atoi(argv[1]);
  if (type_mode < 0 || type_mode > 3) {fprintf(stderr, "ERROR: type_mode must be between 0 and 3"); exit(-1);}

  n_threads = atoi(argv[2]);
  if (n_threads < 1) {fprintf(stderr, "ERROR: n_threads must be greater than 0"); exit(-1);}
  printf("threads: %d\n", n_threads);

  n_iter = atoi(argv[3]);
  if (n_iter < 1) {fprintf(stderr, "ERROR: n_iter must be greater than 0"); exit(-1);}
  printf("iter per thread: %d\n", n_iter);
}