#!/usr/bin/env python3

"""
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
"""


import os
import util
import argparse
import logging

import build
import run_tests

omp = [
    "./codes/omp/omp_atomicadd_scalar.cpp",
    "./codes/omp/omp_atomicadd_array_v2.cpp",
    "./codes/omp/omp_atomicadd_array_v2_close.cpp",
    "./codes/omp/omp_atomicadd_array_v2_spread.cpp",
    #
    "./codes/omp/omp_criticaladd_scalar_spread.cpp",
    #
    "./codes/omp/omp_atomicwrite_scalar.cpp",
    "./codes/omp/omp_atomicread_close.cpp",
    "./codes/omp/omp_atomicread_spread.cpp",
    #
    "./codes/omp/omp_atomiccaptureincrement.cpp",
    #
    "./codes/omp/omp_flush_close.cpp",
    "./codes/omp/omp_flush_spread.cpp",
    #
    "./codes/omp/omp_flush_array_close.cpp",
    "./codes/omp/omp_flush_array_spread.cpp",
    #
    "./codes/omp/omp_barrier.cpp",
    "./codes/omp/omp_barrier_close.cpp",
    "./codes/omp/omp_barrier_spread.cpp",
]

cuda = [
    "./codes/cuda/cuda_atomicadd_scalar_v2.cu",
    "./codes/cuda/cuda_atomicadd_array_v2.cu",
    #
    "./codes/cuda/cuda_atomiccas_array_fail.cu",
    "./codes/cuda/cuda_atomiccas_array_pass.cu",
    "./codes/cuda/cuda_atomiccas_array_pass_v2.cu",
    #
    "./codes/cuda/cuda_atomiccas_scalar_fail.cu",
    "./codes/cuda/cuda_atomiccas_scalar_pass.cu",
    "./codes/cuda/cuda_atomiccas_scalar_pass_v2.cu",
    #
    "./codes/cuda/cuda_atomicexch_v2.cu",
    #
    "./codes/cuda/cuda_threadfence_array.cu",
    "./codes/cuda/cuda_threadfence_block_array.cu",
    "./codes/cuda/cuda_threadfence_system_array.cu",
    #
    "./codes/cuda/cuda_shfl_sync.cu",
    "./codes/cuda/cuda_shfl_up_sync.cu",
    "./codes/cuda/cuda_shfl_down_sync.cu",
    "./codes/cuda/cuda_shfl_xor_sync.cu",
    #
    "./codes/cuda/cuda_syncwarp.cu",
    #
    "./codes/cuda/cuda_syncthreads.cu",
    #
    "./codes/cuda/cuda_match_any_sync.cu",
    "./codes/cuda/cuda_match_all_sync.cu",
    #
    "./codes/cuda/cuda_vote_all_sync.cu",
    "./codes/cuda/cuda_vote_any_sync.cu",
    "./codes/cuda/cuda_vote_ballot_sync.cu",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "codes",
        nargs="+",
        help="list of codes to run (e.g. './codes/omp/omp_atomicadd_scalar.cpp ...'), OR a keyword that corresponds to a set of codes (i.e., 'openmp', 'cuda', or 'all')",
    )
    args = parser.parse_args()

    if len(args.codes) > 1:
        codes = args.codes
    else:
        mode = args.codes[0]
        if mode.lower() == "openmp" or mode.lower() == "omp":
            codes = omp
        elif mode.lower() == "cuda":
            codes = cuda
        elif mode.lower() == "all":
            codes = omp + cuda
        else:
            # fallback to single code
            codes = [mode]

    print(str(util.MachineSpecs()))
    print()
    print("codes to run:")
    for code in codes:
        print(f"  {code}")
    print()
    if input("continue (y/N)? ").lower() != "y":
        exit()

    for code in codes:
        # invoke the test in a seperate python process to allow for redefinition of logger
        os.system(f"./run_tests.py {code}")
