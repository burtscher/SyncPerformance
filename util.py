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
import subprocess
import socket
import logging

type_modes = ["int", "ull", "float", "double"]


class MachineSpecs:
    class HostSpecs:
        def __init__(self):
            self.model_name: str = " ".join(
                capture_stdout("lscpu | grep 'Model name:'").split("\n")[0].split()[2:]
            )
            self.architecture: str = (
                capture_stdout("lscpu | grep 'Architecture'").split("\n")[0].split()[-1]
            )
            self.op_modes: str = " ".join(
                capture_stdout("lscpu | grep 'CPU op-mode(s)'")
                .split("\n")[0]
                .split()[2:]
            )
            self.n_sockets: int = int(
                capture_stdout("lscpu | grep 'Socket(s):'").split("\n")[0].split()[-1]
            )
            self.n_cores_per_socket: int = int(
                capture_stdout("lscpu | grep 'Core(s) per socket:'")
                .split("\n")[0]
                .split()[-1]
            )
            self.n_physical_cores: int = self.n_sockets * self.n_cores_per_socket
            self.n_threads_per_core: int = int(
                capture_stdout("lscpu | grep 'Thread(s) per core:'")
                .split("\n")[0]
                .split()[-1]
            )
            self.n_logical_cores: int = int(
                capture_stdout("lscpu | grep 'CPU(s):'").split("\n")[0].split()[-1]
            )
            self.cache_line_size: int = int(
                capture_stdout(
                    "cat /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
                )
            )
            self.n_numa_nodes: int = 0
            ret = capture_stdout("lscpu | grep 'NUMA node(s):'")
            if ret != "":
                self.n_numa_nodes: int = int(ret.split("\n")[0].split()[-1])

        def __str__(self) -> str:
            return (
                f"CPU: {self.model_name}"
                + f"\n  architecture: {self.architecture}"
                + f"\n  op-modes: {self.op_modes}"
                + f"\n  sockets: {self.n_sockets}"
                + f"\n  cores per socket: {self.n_cores_per_socket}"
                + f"\n  physical cores: {self.n_physical_cores}"
                + f"\n  threads per core: {self.n_threads_per_core}"
                + f"\n  logical cores: {self.n_logical_cores}"
                + f"\n  cache line size (bytes): {self.cache_line_size}"
                + f"\n  NUMA nodes: {self.n_numa_nodes}"
            )

    class DeviceSpecs:
        def __init__(self):
            if not os.path.exists("./check_cuda_props.out"):
                os.system(
                    "nvcc check_cuda_props.cu -O3 -arch=sm_70 -Iinclude -o check_cuda_props.out"
                )
            if not os.path.exists("./check_cuda_props.out"):
                sys.exit("ERROR: './check_cuda_props.cu' failed to compile")
            lines = capture_stdout("./check_cuda_props.out").split("\n")

            self.model_name: str = " ".join(lines[0].split()[1:])
            self.n_SMs: int = int(lines[1].split()[-1])
            self.max_threads_per_SM: int = int(lines[2].split()[-1])
            self.max_blocks: int = int(lines[3].split()[-1])
            self.cores_per_SM: int = int(lines[4].split()[-1])
            self.clock_rate: float = float(lines[5].split()[-1])
            self.regs_per_SM: int = int(lines[6].split()[-1])
            self.regs_per_block: int = int(lines[7].split()[-1])
            self.cache_line_size: int = (
                128  # NOTE: apparently static across all NVIDIA GPUs
            )

        def __str__(self) -> str:
            return (
                f"GPU: {self.model_name}"
                + f"\n  SMs: {self.n_SMs}"
                + f"\n  mTpSM: {self.max_threads_per_SM}"
                + f"\n  max blocks: {self.max_blocks}"
                + f"\n  cores per SM: {self.cores_per_SM}"
                + f"\n  regs per SM: {self.regs_per_SM}"
                + f"\n  regs per block: {self.regs_per_block}"
                + f"\n  clock rate (GHz): {self.clock_rate / 1000 / 1000}"
                + f"\n  cache line size (bytes): {self.cache_line_size}"
            )

    def __init__(self):
        self.hostname = socket.gethostname().split(".")[0]
        self.host = self.HostSpecs()
        self.device = self.DeviceSpecs()

    def __str__(self) -> str:
        return f"system: {self.hostname}\n{self.host}\n{self.device}"

    def configure_strides(self, exe_name: str) -> list:
        if "array" in exe_name:
            if "cuda" in exe_name:
                cache_line_size = self.device.cache_line_size
            else:
                cache_line_size = self.host.cache_line_size
            return [
                1,
                2,
                4,
                cache_line_size // 8,
                cache_line_size // 4,
                cache_line_size,
            ]
        return [0]


def capture_stdout(cmd: str):
    return subprocess.run(
        [cmd],
        capture_output=True,
        shell=True,
        encoding="ascii",
    ).stdout


def gen_base_path(specs: MachineSpecs, exe_name: str) -> str:
    """Generate the base results path up to code directory (no arr step subdir)."""
    path = os.path.join(
        f"results",
        specs.hostname,
        "_".join(specs.device.model_name.split()) if "cuda" in exe_name else "",
        exe_name,
    )
    return path


def gen_results_path(specs: MachineSpecs, exe_name: str, arr_step: int) -> str:
    path = os.path.join(
        gen_base_path(specs, exe_name),
        f"step{arr_step}" if "array" in exe_name else "",
    )
    if not os.path.exists(path):
        os.makedirs(path)

    # also build aggregate subdirectories if they do not exist
    # subdirs = ["max", "min", "med", "avg"]
    # for subdir in subdirs:
    #     subdir_path = os.path.join(path, subdir)
    #     if not os.path.exists(subdir_path):
    #         os.makedirs(subdir_path)

    return path


def add_log_level(
    level_name: str, level_num: int = logging.INFO + 5, method_name: str = None
):
    """
    https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility/35804945#35804945

    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5
    """
    if not method_name:
        method_name = level_name.lower()

    if hasattr(logging, level_name):
        raise AttributeError("{} already defined in logging module".format(level_name))
    if hasattr(logging, method_name):
        raise AttributeError("{} already defined in logging module".format(method_name))
    if hasattr(logging.getLoggerClass(), method_name):
        raise AttributeError("{} already defined in logger class".format(method_name))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), method_name, logForLevel)
    setattr(logging, method_name, logToRoot)


def setup_logger(specs: MachineSpecs, exe_name: str):
    # create directories
    log_path = gen_base_path(specs, exe_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # clear previously-existing log
    log_file = os.path.join(log_path, "log.txt")
    if os.path.exists(log_file):
        os.system(f"rm {log_file}")

    add_log_level("OK")
    format = "%(message)s"
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=logging.OK, format=format, handlers=handlers)
