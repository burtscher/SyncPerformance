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


/*
 * This header file includes global constants used for test configuration.
 * It is also where certain debugging helpers are defined/initialized.
*/

#include <fstream>

// number of individual operations to repeat within each "iter" any loop bound by this value should have an accompanying "#pragma unroll"
#define N_UNROLL 100

// number of warmup rounds to perform before timing
#define N_WARMUP 1

// number of runs per kernel/func from which to collect the median
#define N_RUNS 9

// number of times a single "run" can attempt a retry in the event of a negative reading the entire test is aborted if this limit is reached
#define N_ATTEMPTS 7

/*
 * Some tests measure more than one repetition of an op.
 * We define this variable to be able to adjust this per test, and still have accurate timings.
 * 
 * NOTE: this should be the number of ops after the "difference" between the two functions/kernels.
 * i.e., func1(){op;} and func2(){op; op; op;} should set N_OPS_PER_TEST_UNIT = 2.
*/
int N_OPS_PER_TEST_UNIT = 1;

// NOTE: uncomment this to enable fine grain results when compiling a code (called in run_test())
// #define FINE_GRAIN_RESULTS_MODE 1

#ifdef FINE_GRAIN_RESULTS_MODE

void record_all_thread_runtimes_omp(double* timers1, double* timers2, const int n_timers, const int n_threads, const int n_iter, const int arr_step);
void record_all_thread_runtimes_omp(double* timers1, double* timers2, const int n_timers, const int n_threads, const int n_iter);
void print_fine_grain_header_omp(std::ofstream& outfile, const int n_threads, const int n_iter, const int arr_step);
void print_fine_grain_header_omp(std::ofstream& outfile, const int n_threads, const int n_iter);
void record_all_thread_runtimes_cuda(long long* timers1, long long* timers2, const int n_timers, const int n_threads, const int n_blocks, const int n_iter, const int arr_step);
void record_all_thread_runtimes_cuda(long long* timers1, long long* timers2, const int n_timers, const int n_threads, const int n_blocks, const int n_iter);
void print_fine_grain_header_cuda(std::ofstream& outfile, const int n_threads, const int n_blocks, const int n_iter, const int arr_step);
void print_fine_grain_header_cuda(std::ofstream& outfile, const int n_threads, const int n_blocks, const int n_iter);
void print_runtimes_all_threads(std::ofstream& outfile, double* timers1, double* timers2, const int n_timers, const int n_iter);
void print_cycles_all_threads(std::ofstream& outfile, long long* timers1, long long* timers2, const int n_timers, const int n_iter);

void record_all_thread_runtimes_omp(double* timers1, double* timers2, const int n_timers, const int n_threads, const int n_iter, const int arr_step) {
  std::ofstream outfile;
  outfile.open("./all_thread_runtimes.csv", std::ofstream::out);

  print_fine_grain_header_omp(outfile, n_threads, n_iter, arr_step);
  print_runtimes_all_threads(outfile, timers1, timers2, n_timers, n_iter);

  outfile.close();
}

void record_all_thread_runtimes_omp(double* timers1, double* timers2, const int n_timers, const int n_threads, const int n_iter) {
  std::ofstream outfile;
  outfile.open("./all_thread_runtimes.csv", std::ofstream::out);

  print_fine_grain_header_omp(outfile, n_threads, n_iter);
  print_runtimes_all_threads(outfile, timers1, timers2, n_timers, n_iter);

  outfile.close();
}

void print_fine_grain_header_omp(std::ofstream& outfile, const int n_threads, const int n_iter, const int arr_step) {
  outfile << "n_threads," << n_threads << std::endl;
  outfile << "n_iter," << n_iter << std::endl;
  outfile << "arr_step," << arr_step << std::endl;
  outfile << std::endl;
}

void print_fine_grain_header_omp(std::ofstream& outfile, const int n_threads, const int n_iter) {
  outfile << "n_threads," << n_threads << std::endl;
  outfile << "n_iter," << n_iter << std::endl;
  outfile << std::endl;
}

void record_all_thread_runtimes_cuda(long long* timers1, long long* timers2, const int n_timers, const int n_threads, const int n_blocks, const int n_iter, const int arr_step) {
  std::ofstream outfile;
  outfile.open("./all_thread_runtimes.csv", std::ofstream::out);

  print_fine_grain_header_cuda(outfile, n_threads, n_blocks, n_iter, arr_step);
  print_cycles_all_threads(outfile, timers1, timers2, n_timers, n_iter);

  outfile.close();
}

void record_all_thread_runtimes_cuda(long long* timers1, long long* timers2, const int n_timers, const int n_threads, const int n_blocks, const int n_iter) {
  std::ofstream outfile;
  outfile.open("./all_thread_runtimes.csv", std::ofstream::out);

  print_fine_grain_header_cuda(outfile, n_threads, n_blocks, n_iter);
  print_cycles_all_threads(outfile, timers1, timers2, n_timers, n_iter);

  outfile.close();
}

void print_fine_grain_header_cuda(std::ofstream& outfile, const int n_threads, const int n_blocks, const int n_iter, const int arr_step) {
  outfile << "n_threads," << n_threads << std::endl;
  outfile << "n_blocks," << n_blocks << std::endl;
  outfile << "n_iter," << n_iter << std::endl;
  outfile << "arr_step," << arr_step << std::endl;
  outfile << std::endl;
}

void print_fine_grain_header_cuda(std::ofstream& outfile, const int n_threads, const int n_blocks, const int n_iter) {
  outfile << "n_threads," << n_threads << std::endl;
  outfile << "n_blocks," << n_blocks << std::endl;
  outfile << "n_iter," << n_iter << std::endl;
  outfile << std::endl;
}

void print_runtimes_all_threads(std::ofstream& outfile, double* timers1, double* timers2, const int n_timers, const int n_iter) {
  outfile << "tid,runtime" << std::endl;
  for (int i = 0; i < n_timers; i++) {
    outfile << i << "," << ((timers2[i] - timers1[i]) / (double)n_iter / (double)N_UNROLL) << std::endl;
  }
}

void print_cycles_all_threads(std::ofstream& outfile, long long* timers1, long long* timers2, const int n_timers, const int n_iter) {
  outfile << "tid,cycles" << std::endl;
  for (int i = 0; i < n_timers; i++) {
    outfile << i << "," << ((timers2[i] - timers1[i]) / (long long)n_iter / (long long)N_UNROLL) << std::endl;
  }
}

#endif /* FINE_GRAIN_RESULTS_MODE */