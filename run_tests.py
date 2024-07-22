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
import numpy as np
import csv
import pickle
import argparse
import logging

import build
import util
import draw_graph


def execute_omp(exe_name: str, specs: util.MachineSpecs):
    type_range = range(len(util.type_modes))
    if "omp_barrier" in exe_name:
        type_range = [0]
    thread_range = range(1, specs.host.n_logical_cores + 1 + 4)
    n_iter = 1000

    strides = specs.configure_strides(exe_name)
    for stride in strides:
        results_path = util.gen_results_path(specs, exe_name, stride)
        max_per_op = np.zeros((len(type_range), len(thread_range)))
        min_per_op = np.zeros((len(type_range), len(thread_range)))
        med_per_op = np.zeros((len(type_range), len(thread_range)))
        avg_per_op = np.zeros((len(type_range), len(thread_range)))
        for i in range(len(type_range)):
            type_mode = type_range[i]
            for j in range(len(thread_range)):
                n_threads = thread_range[j]
                if "array" in exe_name:
                    cmd = f"./{exe_name} {type_mode} {n_threads} {n_iter} {stride}"
                else:
                    cmd = f"./{exe_name} {type_mode} {n_threads} {n_iter}"
                if "atomicread" in exe_name:
                    cmd += f" 12"
                logging.ok(cmd)
                stdout = util.capture_stdout(cmd)
                logging.ok(stdout)
                lines = stdout.split("\n")
                for line in lines:
                    if "max per op:" in line:
                        max_per_op[i, j] = float(line.split()[-1])
                    if "min per op:" in line:
                        min_per_op[i, j] = float(line.split()[-1])
                    if "med per op:" in line:
                        med_per_op[i, j] = float(line.split()[-1])
                    if "avg per op:" in line:
                        avg_per_op[i, j] = float(line.split()[-1])

        # write to binary
        bin_filepath = f"{results_path}/runtimes.bin"
        with open(bin_filepath, "wb") as outfile:
            export_data = (
                specs,
                exe_name,
                type_range,
                thread_range,
                n_iter,
                max_per_op,
                min_per_op,
                med_per_op,
                avg_per_op,
            )
            pickle.dump(export_data, outfile)

        # write to csv
        csv_filepath = f"{results_path}/runtimes.csv"
        with open(csv_filepath, "w") as outfile:
            writer = csv.writer(outfile)

            # header
            writer.writerow([exe_name])
            writer.writerow(["n iter", f"{n_iter}"])
            if "array" in exe_name:
                writer.writerow(["arr step", f"{stride}"])
            writer.writerow([""])

            # table
            writer.writerow(
                ["type mode", "aggregate"] + [str(item) for item in thread_range]
            )
            for i in type_range:
                type_mode = util.type_modes[i]
                writer.writerow(
                    [f"{type_mode}", "max"] + [str(item) for item in max_per_op[i, :]]
                )
                writer.writerow(
                    [f"{type_mode}", "min"] + [str(item) for item in min_per_op[i, :]]
                )
                writer.writerow(
                    [f"{type_mode}", "med"] + [str(item) for item in med_per_op[i, :]]
                )
                writer.writerow(
                    [f"{type_mode}", "avg"] + [str(item) for item in avg_per_op[i, :]]
                )

        draw_graph.handle_omp_graphs(results_path)

    # delete executable
    if os.path.exists(exe_name):
        os.system(f"rm {exe_name}")
    else:
        sys.exit("ERROR: deleting executable failed (compilation failed?)")

    logging.ok(f"done, results and figures in '{util.gen_base_path(specs, exe_name)}'")


def execute_cuda(exe_name: str, specs: util.MachineSpecs):
    type_range = range(len(util.type_modes))
    if "atomiccas" in exe_name:
        type_range = [0, 1]
    if "atomicexch" in exe_name:
        type_range = [0, 1, 2]
    if (
        "reduce_add_sync" in exe_name
        or "syncthreads" in exe_name
        or "syncwarp" in exe_name
        or "vote" in exe_name  # all_sync, any_sync, ballot_sync
    ):
        type_range = [0]
    thread_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    block_range = [
        1,
        2,
        specs.device.n_SMs // 2,
        specs.device.n_SMs,
        specs.device.n_SMs * 2,
    ]
    if specs.device.max_blocks > (2 * specs.device.n_SMs):
        block_range.append(specs.device.max_blocks)  # for GPUs w/ mTpSM >= 2 * 1024
    n_iter = 1000

    for arr_step in specs.configure_strides(exe_name):
        results_path = util.gen_results_path(specs, exe_name, arr_step)
        max_per_op = np.zeros((len(type_range), len(thread_range), len(block_range)))
        min_per_op = np.zeros((len(type_range), len(thread_range), len(block_range)))
        med_per_op = np.zeros((len(type_range), len(thread_range), len(block_range)))
        avg_per_op = np.zeros((len(type_range), len(thread_range), len(block_range)))
        for i in range(len(type_range)):
            type_mode = type_range[i]
            for j in range(len(thread_range)):
                n_threads = thread_range[j]
                for k in range(len(block_range)):
                    n_blocks = block_range[k]
                    if "array" in exe_name:
                        cmd = f"./{exe_name} {type_mode} {n_threads} {n_blocks} {n_iter} {arr_step}"
                    else:
                        cmd = (
                            f"./{exe_name} {type_mode} {n_threads} {n_blocks} {n_iter}"
                        )
                    logging.ok(cmd)
                    stdout = util.capture_stdout(cmd)
                    logging.ok(stdout)
                    lines = stdout.split("\n")
                    for line in lines:
                        if "max per op:" in line:
                            max_per_op[i, j, k] = (
                                float(line.split()[-1]) / specs.device.clock_rate
                            )
                        if "min per op:" in line:
                            min_per_op[i, j, k] = (
                                float(line.split()[-1]) / specs.device.clock_rate
                            )
                        if "med per op:" in line:
                            med_per_op[i, j, k] = (
                                float(line.split()[-1]) / specs.device.clock_rate
                            )
                        if "avg per op:" in line:
                            avg_per_op[i, j, k] = (
                                float(line.split()[-1]) / specs.device.clock_rate
                            )

        # write to binary
        bin_filepath = f"{results_path}/runtimes.bin"
        with open(bin_filepath, "wb") as outfile:
            export_data = (
                specs,
                exe_name,
                type_range,
                thread_range,
                block_range,
                n_iter,
                max_per_op,
                min_per_op,
                med_per_op,
                avg_per_op,
            )
            pickle.dump(export_data, outfile)

        # write to csv
        csv_filepath = f"{results_path}/runtimes.csv"
        with open(csv_filepath, "w") as outfile:
            writer = csv.writer(outfile)

            # header
            writer.writerow([exe_name])
            writer.writerow(["n iter", f"{n_iter}"])
            if "array" in exe_name:
                writer.writerow(["arr step", f"{arr_step}"])

            # tables
            for k in range(len(block_range)):
                n_blocks = block_range[k]
                writer.writerow([""])
                writer.writerow(["n_blocks", f"{n_blocks}"])
                writer.writerow(
                    ["type mode", "aggregate"] + [str(item) for item in thread_range]
                )
                for i in type_range:
                    type_mode = util.type_modes[i]
                    writer.writerow(
                        [f"{type_mode}", "max"]
                        + [str(item) for item in max_per_op[i, :, k]]
                    )
                    writer.writerow(
                        [f"{type_mode}", "min"]
                        + [str(item) for item in min_per_op[i, :, k]]
                    )
                    writer.writerow(
                        [f"{type_mode}", "med"]
                        + [str(item) for item in med_per_op[i, :, k]]
                    )
                    writer.writerow(
                        [f"{type_mode}", "avg"]
                        + [str(item) for item in avg_per_op[i, :, k]]
                    )

        draw_graph.handle_cuda_graphs(results_path)

    # delete executable
    if os.path.exists(exe_name):
        os.system(f"rm {exe_name}")
    else:
        sys.exit("ERROR: deleting executable failed (compilation failed?)")

    logging.ok(f"done, results and figures in '{util.gen_base_path(specs, exe_name)}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("code_path")
    args = parser.parse_args()

    specs = util.MachineSpecs()

    filename = os.path.basename(args.code_path)
    exe_name = filename.split(".")[0]

    util.setup_logger(specs, exe_name)

    logging.ok("SyncPerformance v0.1")
    logging.ok(f"code: {filename}")
    logging.ok(str(util.MachineSpecs()))

    build.compile(args.code_path, exe_name)
    if "omp" in exe_name:
        execute_omp(exe_name, specs)
    elif "cuda" in exe_name:
        execute_cuda(exe_name, specs)
