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


import sys
import os
import pickle
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import numpy as np

import util

# toggle
SHORT_GRAPH_MODE = False
# SHORT_GRAPH_MODE = True

FIGSIZE = (4, 2.5)
if SHORT_GRAPH_MODE:
    FIGSIZE = (4, 1.75)

marker_shapes = ["o", "s", "^", "d"]

my_color_palette = ["#0039b2", "#00b9be", "#ffb200", "#c1282d"]


def set_style():
    rc_params = {}
    sns.set_theme(rc=rc_params)
    sns.set_context("paper")  # controls font scaling


def handle_omp_graphs(results_path):
    set_style()

    if not os.path.exists(results_path):
        sys.exit(f"ERROR: path '{results_path}' does not exist")

    bin_filepath = f"{results_path}/runtimes.bin"
    with open(bin_filepath, "rb") as infile:
        (
            specs,
            exe_name,
            type_range,
            thread_range,
            n_iter,
            max_per_op,
            min_per_op,
            med_per_op,
            avg_per_op,
        ) = pickle.load(infile)

    def draw_omp_graph(data, aggregate_type: str):
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # remove result at 1 thread (irrelevant to parallel performance, since no syncs happen)
        x_min_limit = 2 - 1
        x_max_limit = specs.host.n_logical_cores
        x_val = thread_range[x_min_limit : x_max_limit + 1]
        data = data[:, x_min_limit : x_max_limit + 1]

        for i in type_range:
            type_mode = util.type_modes[i]
            y_val = 1 / data[i, :]  # throughput
            for j in range(len(data[i, :])):
                if data[i, j] <= 0:
                    if aggregate_type == "max":
                        print(
                            f"WARNING: negative runtime @ {aggregate_type}[type={type_mode}, thread={thread_range[j]}]"
                        )
            plt.plot(
                x_val,
                y_val,
                label=f"{type_mode}",
                color=my_color_palette[i],
                linestyle="dashed" if i % 2 else "-",
                linewidth=2,
            )

        # plt.title(f"{exe_name}, {aggregate_type}")
        plt.xlabel("Threads")
        plt.xlim(left=x_min_limit + 1, right=x_max_limit)
        plt.ylim(bottom=0)  # 0-based y-axis

        # bespoke adjustments for figures in paper
        # if "atomicadd_array_v2" in exe_name and "ithaca" in specs.hostname:
        #     plt.ylim(bottom=0, top=125_000_000)
        # if "flush_array_" in exe_name and "austin" in specs.hostname:
        #     plt.ylim(bottom=0, top=(3.5 * 10**7))
        # if "flush_array_" in exe_name and "austin" in specs.hostname:
        #     plt.ylim(bottom=0, top=(3.5 * 10**8))

        # move scientific notation from corner to y-axis label
        ax.ticklabel_format(useMathText=True)
        fig.canvas.draw()  # refreshes the canvas to generate the tick label
        ax.yaxis.offsetText.set_visible(False)
        offset = ax.yaxis.get_major_formatter().get_offset()
        ax.yaxis.set_label_text(f"Ops. per Sec. per Thread ({offset})")
        if SHORT_GRAPH_MODE:
            ax.yaxis.set_label_text(f"Ops. per Sec.\nper Thread ({offset})")

        # draw legend if we have more than one type
        if len(type_range) > 1:
            ax.legend(
                loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=len(type_range)
            )

        x_tick_range = list(
            range(8, x_max_limit + 1, 8)
            if x_max_limit > 32
            else range(4, x_max_limit + 1, 4)
        )
        x_tick_range = [x_min_limit + 1] + x_tick_range
        ax.tick_params(pad=-1)
        plt.xticks(ticks=x_tick_range)

        plt.axvline(
            specs.host.n_physical_cores, color="k", linestyle="--"
        )  # draw vertical line number of physical cores

        # plt.savefig(
        #     f"{results_path}/{aggregate_type}/{exe_name}.pdf", bbox_inches="tight"
        # )
        if aggregate_type == "max":
            plt.savefig(f"{results_path}/{exe_name}.pdf", bbox_inches="tight")
        plt.close()

    draw_omp_graph(max_per_op, "max")
    # draw_omp_graph(min_per_op, "min")
    # draw_omp_graph(med_per_op, "med")
    # draw_omp_graph(avg_per_op, "avg")


def handle_cuda_graphs(results_path):
    set_style()

    if not os.path.exists(results_path):
        sys.exit(f"ERROR: path '{results_path}' does not exist")

    bin_filepath = f"{results_path}/runtimes.bin"
    with open(bin_filepath, "rb") as infile:
        (
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
        ) = pickle.load(infile)

    index786 = -1
    if 786 in thread_range:
        # remove incorrect thread value if it exists (for older tests where I made a typo)
        index786 = thread_range.index(786)
        thread_range.remove(786)

    def draw_cuda_graph_per_n_blocks(data, aggregate_type: str):
        if index786 != -1:
            data = np.delete(data, index786, axis=1)
        for k in range(len(block_range)):
            fig, ax = plt.subplots(figsize=FIGSIZE)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            n_blocks = block_range[k]
            x_val = range(
                1, len(thread_range) + 1
            )  # for equidistant, readable plotting

            for i in type_range:
                type_mode = util.type_modes[i]
                y_val = 1 / data[i, :, k]  # throughput
                for j in range(len(data[i, :, k])):
                    if data[i, j, k] <= 0:
                        if aggregate_type == "max":
                            print(
                                f"WARNING: negative runtime @ {aggregate_type}[type={type_mode}, thread={thread_range[j]}, block={block_range[k]}]"
                            )
                plt.plot(
                    x_val,
                    y_val,
                    label=f"{type_mode}",
                    marker=marker_shapes[i],
                    fillstyle="none" if i % 2 else "full",
                    linestyle="dashed" if i % 2 else "-",
                    color=my_color_palette[i],
                    linewidth=2,
                )

            # plt.title(f"{exe_name}, n_blocks={n_blocks}, {aggregate_type}")
            plt.xlabel("Threads per Block")
            plt.xlim(left=list(x_val)[0], right=list(x_val)[-1])

            # bespoke adjustments for figures in paper
            # if "cuda_atomicadd_scalar_v2" in exe_name and "ithaca" in specs.hostname:
            #     plt.ylim(bottom=0, top=(4.45 * 10**5))
            # if "cuda_atomicadd_array_v2" in exe_name and "ithaca" in specs.hostname:
            #     plt.ylim(bottom=0, top=(4.45 * 10**5))
            # if "cuda_atomiccas" in exe_name and "ithaca" in specs.hostname:
            #     plt.ylim(bottom=0, top=(3.5 * 10**5))
            # if "cuda_atomicexch" in exe_name and "ithaca" in specs.hostname:
            #     plt.ylim(bottom=0, top=(4.5 * 10**5))
            # if "cuda_threadfence_array" in exe_name and "ithaca" in specs.hostname:
            #     plt.ylim(bottom=0, top=(6.1 * 10**3))

            # move scientific notation from corner to y-axis label
            ax.ticklabel_format(useMathText=True)
            fig.canvas.draw()  # refreshes the canvas to generate the tick label
            ax.yaxis.offsetText.set_visible(False)
            offset = ax.yaxis.get_major_formatter().get_offset()
            ax.yaxis.set_label_text(f"Ops. per Sec. per Thread ({offset})")
            if SHORT_GRAPH_MODE:
                ax.yaxis.set_label_text(f"Ops. per Sec.\nper Thread ({offset})")

            ax.tick_params(pad=-1)
            plt.xticks(ticks=x_val, labels=thread_range, rotation=45)

            # draw legend if we have more than one type
            if len(type_range) > 1:
                ax.legend(
                    loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=len(type_range)
                )
            # plt.savefig(
            #     f"{results_path}/{aggregate_type}/{exe_name}_{n_blocks}blocks.pdf",
            #     bbox_inches="tight",
            # )
            if aggregate_type == "max":
                plt.savefig(
                    f"{results_path}/{exe_name}_{n_blocks}blocks.pdf",
                    bbox_inches="tight",
                )
            plt.close()

    draw_cuda_graph_per_n_blocks(max_per_op, "max")
    # draw_cuda_graph_per_n_blocks(min_per_op, "min")
    # draw_cuda_graph_per_n_blocks(med_per_op, "med")
    # draw_cuda_graph_per_n_blocks(avg_per_op, "avg")


def get_immediate_subdirectories(a_dir):
    return [
        os.path.join(a_dir, name)
        for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))
    ]


def draw_graphs(path):
    if "omp" in path:
        handle_omp_graphs(path)
    elif "cuda" in path:
        handle_cuda_graphs(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_paths",
        nargs="+",
        help="List of paths to a test's results directory, e.g., './results/<system>/<omp_test>/' or './results/<system>/<gpu>/<cuda_test>/'",
    )
    args = parser.parse_args()

    for path in args.results_paths:
        print(path)
        subpaths = []
        if "NVIDIA" in os.path.basename(path):
            print("  skipping...")
            continue  # skip GPU if it's caught wildcarding the CPU tests
        if "array" in path:
            subpaths = get_immediate_subdirectories(path)
        else:
            subpaths = [path]
        for p in subpaths:
            if p != path:
                print(f"  {p}")
            draw_graphs(p)
