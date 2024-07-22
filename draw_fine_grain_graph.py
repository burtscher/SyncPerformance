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


import argparse
import csv
import matplotlib.pyplot as plt


def draw_fine_grain_results(results_csv, output_path):
    x_axis_label = ""
    data = []

    with open(results_csv, "r") as infile:
        reader = csv.reader(infile)
        for row in reader:
            if len(row) < 1:
                continue
            if "tid" in row[0]:
                x_axis_label = row[1]  # runtime or cycles
                break  # skip to results
        for row in reader:
            data.append(row[1])

    # readable tick marks
    # xticks = []
    # if len(data) < 32:
    #     xticks = range(len(data), 2)
    # elif len(data) < 128:
    #     xticks = range(len(data), 8)
    # else:
    #     xticks = range(len(data), 32)

    fig, ax = plt.subplots()
    ax.bar(range(len(data)), data)
    # plt.xticks(ticks=xticks, labels=xticks)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o")
    args = parser.parse_args()

    draw_fine_grain_results("./all_thread_runtimes.csv", args.output)