# SyncPerformance

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13227900.svg)](https://doi.org/10.5281/zenodo.13227900)

This repository hosts a suite of codes that measure the execution time of single synchronization primitives from CUDA and OpenMP. A full description of the test templates and methodology can be found in our paper (see below).

If you use any of the code or results in this repository, please cite the following publication:

>Brandon Alexander Burtchell and Martin Burtscher. "Characterizing CUDA and OpenMP Synchronization Primitives." Proceedings of the IEEE International Symposium on Workload Characterization. September 2024.

## Installation and Setup

The Python scripts that automate compilation and running require the following packages. We recommend installing and managing these with [pip](https://pypi.org/project/pip/):

```bash
pip3 install numpy matplotlib seaborn
```

If testing CUDA codes, you may want to specify the compute capability of the compiled codes to match your GPU. To do so, create `./config.py` in the root of the repository. Inside, specify the desired `nvcc` "`-arch=`" argument ([more info](https://developer.nvidia.com/cuda-gpus)). For example, for an NVIDIA GeForce RTX 4090, the latest supported compute capability is 8.9:

```python
nvcc_arch = "sm_89"
```

A working example is provided in `./config.py.example` and can be copied. If the compiled compute capability isn't a concern (e.g., if you just want to run OpenMP codes), this step can be ignored (the script will default to `nvcc_arch = "native"`)

If your system has multiple GPUs, ensure the GPU of interest is selected before running any scripts, e.g.:

```bash
export CUDA_VISIBLE_DEVICES=1
```

## Running Codes

To run all codes:

```bash
./launch.py all
```

To run only the OpenMP or CUDA set of codes, specify which as an argument. For example, to only run the CUDA codes:

```bash
./launch.py cuda
```

To test individual codes across all parameters, list the paths to the desired codes. For example:

```bash
./launch.py ./codes/omp/omp_atomicadd_scalar.cpp ./codes/cuda/cuda_syncwarp.cu
```

Each test's results and figures will be output to a corresponding directory in `./results/<hostname>/`.

## Experiment Customization

Global test parameters (e.g., `N_UNROLL`, `N_RUNS`) can be changed in `./include/config.h`. Other per-code test parameters (e.g., `n_iter`, `thread_range`) can be changed in `./run_tests.py` for OpenMP and CUDA codes in their respective functions (`execute_omp()` and `execute_cuda()`).

## Paper Results and Figures

The raw results and figures from our three tested systems are included in the `./results/system*/` directories.

## Miscellaneous Scripts

To regenerate graphs for existing data, run `./draw_graph.py`, passing any number of directory paths that correspond to a code's root results directory. The script will automatically handle tests with varying strides as long as the code has "`array`" in its name.

```bash
./draw_graph.py ./results/*/omp_barrier_*/
```

To compile an individual code to run manually, run the following. Pass the `-a` argument to generate assembly/PTX instead. Pass the `-d` argument to compile the code for debugging.

```bash
./build.py path/to/code
```

Run the resulting executable (generated in the current working directory) without any arguments to recieve a help message that lists the required arguments.

If "`#define FINE_GRAIN_RESULTS_MODE 1`" is uncommented in `./include/config.h`, running a code manually will output `all_thread_runtimes.csv` in the current working directory. This CSV lists the runtime/cycles of a single primitive for each active thread. To generate a figure that visualizes this, run `./draw_fine_grain_graph.py`. Note that we omit mentioning this in the paper. Nonetheless, we provide it here in case if any future researchers discover any interesting findings with this tool.
