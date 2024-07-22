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
import sys
import argparse

try:
    from config import nvcc_arch
except ImportError:
    nvcc_arch = "native"


def compile(code_path: str, exe_name: str):
    if "omp" in code_path:
        ret = os.system(f"g++ {code_path} -O3 -fopenmp -Iinclude -o {exe_name}")
    elif "cuda" in code_path:
        ret = os.system(
            f"nvcc {code_path} -O3 -arch={nvcc_arch} -Iinclude -o {exe_name}"
        )
    else:
        sys.exit(f"ERROR: provided code '{code_path}' was not omp or cuda")

    if ret != 0:
        sys.exit(f"ERROR: compilation of {code_path} failed")


def compile_debug(code_path: str, exe_name: str):
    if "omp" in code_path:
        os.system(f"g++ {code_path} -O3 -fopenmp -Iinclude -g -o {exe_name}")
    elif "cuda" in code_path:
        os.system(f"nvcc {code_path} -O3 -arch={nvcc_arch} -Iinclude -g -o {exe_name}")
    else:
        sys.exit(f"ERROR: provided code '{code_path}' was not omp or cuda")


def compile_assembly(code_path: str, exe_name: str):
    if "omp" in code_path:
        os.system(f"g++ {code_path} -O3 -fopenmp -Iinclude -S -o {exe_name}.asm")
    elif "cuda" in code_path:
        os.system(
            f"nvcc {code_path} -O3 -arch={nvcc_arch} -Iinclude -ptx -o {exe_name}.ptx"
        )
    else:
        sys.exit(f"ERROR: provided code '{code_path}' was not omp or cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("code_path", help="path to source code")
    parser.add_argument(
        "--assembly",
        "-a",
        action="store_true",
        help="generate assembly code instead of executable",
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="compile with debug flag '-g'"
    )
    args = parser.parse_args()

    filename = os.path.basename(args.code_path)
    exe_name = filename.split(".")[0]

    if args.assembly:
        compile_assembly(args.code_path, exe_name)
    elif args.debug:
        compile_debug(args.code_path, exe_name)
    else:
        compile(args.code_path, exe_name)
