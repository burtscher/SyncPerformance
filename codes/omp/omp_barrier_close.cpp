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
#include "omp_test_setup_nopointerparam.h"

int main(int argc, char* argv[]) {
  printf("OMP barrier, proc_bind(close)\n");

  int type_mode, n_threads, n_iter;
  process_args(argc, argv, type_mode, n_threads, n_iter);

  switch (type_mode) {
    case 0:
      run_test<int>(n_threads, n_iter);
      break;
  }

  printf("\n");
  return 0;
}

template <typename T>
static void func1(const int n_threads, const int n_iter, double* const timer) {
  #pragma omp parallel num_threads(n_threads) proc_bind(close) default(none) shared(n_iter, timer)
  {
    for (int i = 0; i < N_WARMUP; i++) {
      #pragma unroll
      for (int j = 0; j < N_UNROLL; j++) {
        #pragma omp barrier
      }
    }

    #pragma omp barrier
    timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < n_iter; i++) {
      #pragma unroll
      for (int j = 0; j < N_UNROLL; j++) {
        #pragma omp barrier
      }
    }

    gettimeofday(&end, NULL);
    timer[omp_get_thread_num()] = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  }
}

template <typename T>
static void func2(const int n_threads, const int n_iter, double* const timer) {
  #pragma omp parallel num_threads(n_threads) proc_bind(close) default(none) shared(n_iter, timer)
  {
    for (int i = 0; i < N_WARMUP; i++) {
      #pragma unroll
      for (int j = 0; j < N_UNROLL; j++) {
        #pragma omp barrier
        #pragma omp barrier
      }
    }

    #pragma omp barrier
    timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < n_iter; i++) {
      #pragma unroll
      for (int j = 0; j < N_UNROLL; j++) {
        #pragma omp barrier
        #pragma omp barrier
      }
    }

    gettimeofday(&end, NULL);
    timer[omp_get_thread_num()] = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  }
}
