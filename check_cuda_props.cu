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


#include <cuda.h>
#include <cstdio>
#include "cuda_macros.cuh"

int getSPcores(cudaDeviceProp devProp);

int main() {
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    fprintf(stderr, "ERROR: no CUDA capable device detected\n\n");
    exit(-1);
  }

  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int blocks = SMs * (mTpSM / 1024);

  printf("GPU: %s\n", deviceProp.name);
  printf("  SMs: %d\n", SMs);
  printf("  mTpSM: %d\n", mTpSM);
  printf("  max blocks: %d\n", blocks);
  printf("  cores per SM: %d\n", getSPcores(deviceProp) / SMs);
  printf("  clock rate: %d\n", deviceProp.clockRate);  // NOTE: deprecated?
  printf("  regs per SM: %d\n", deviceProp.regsPerMultiprocessor);
  printf("  regs per block: %d\n", deviceProp.regsPerBlock);

  CheckCuda();

  return 0;
}

/**
 * https://stackoverflow.com/a/32531982
 */
int getSPcores(cudaDeviceProp devProp) {
  int cores = 0;
  int mp = devProp.multiProcessorCount;
  switch (devProp.major) {
    case 2:  // Fermi
      if (devProp.minor == 1)
        cores = mp * 48;
      else
        cores = mp * 32;
      break;
    case 3:  // Kepler
      cores = mp * 192;
      break;
    case 5:  // Maxwell
      cores = mp * 128;
      break;
    case 6:  // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2))
        cores = mp * 128;
      else if (devProp.minor == 0)
        cores = mp * 64;
      else
        printf("Unknown device type\n");
      break;
    case 7:  // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5))
        cores = mp * 64;
      else
        printf("Unknown device type\n");
      break;
    case 8:  // Ampere
      if (devProp.minor == 0)
        cores = mp * 64;
      else if (devProp.minor == 6)
        cores = mp * 128;
      else if (devProp.minor == 9)
        cores = mp * 128;  // ada lovelace
      else
        printf("Unknown device type\n");
      break;
    case 9:  // Hopper
      if (devProp.minor == 0)
        cores = mp * 128;
      else
        printf("Unknown device type\n");
      break;
    default:
      printf("Unknown device type\n");
      break;
  }
  return cores;
}