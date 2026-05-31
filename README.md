# SYnergy Python binding

This repository contains a Python binding layer for [SYnergy](https://github.com/unisa-hpc/SYnergy/tree/aurora-geopm), integrated with a custom `dpctl` fork. 

The goal of the project is to expose selected SYnergy features to Python while keeping the programming model very close to `dpctl`. 

The binding acts as a thin facade placed over the native SYnergy/SYCL backend: on the surface, Python is used to create devices, queues, submit kernels, inspect profiling results and control device frequencies, delegating the low-level execution to the native SYnergy implementation.

## SYnergy/dpctl connection

The project connects three layers:

```text
Python
    -> dpctl-compatible facade
        -> Cython / pybind11 native bridge
            -> SYnergy / SYCL backend
	
```

The original SYnergy project provides a C++ SYCL library for energy measurement, including per-device and per-kernel energy profiling, and frequency manipulation, applying scaling to both memory and core frequency. 

This fork further extends the library with a Python-facing layer built around the [`dpctl`](https://intelpython.github.io/dpctl/latest/index.html) library, Cython and pybind11. 

## Main Features

The current Python binding exposes:

- creation of SYnergy-aware devices from dpctl selectors;
- creation of SYnergy-aware queues;
device information and backend inspection;
- querying supported core and uncore frequencies;
- reading current frequency values;
- setting core, uncore or combined frequencies;
- submission of existing dpctl.program.SyclKernel objects;
- optional device-level and kernel-level profiling;
- OpenCL source and SPIR-V helper submission paths;
- direct native kernel submission through a factory-based pipeline.


The 2 main public abstraction are: 

```text
SYnergyDevice
    -> device information
    -> frequency management
    -> backend support checks

SYnergyQueue
    -> kernel submission
    -> synchronization
    -> device/kernel profiling
```


## Repository Structure

The most relevant folders included in the repository are: 

```text
include/
    Native SYnergy headers.

binding/
    Native C++/pybind11/Cython binding code.

python/
    Python facade and tests.
	bindings/
		python package for SYnergyDevice and SYnergyQueue

docs/
    MkDocs documentation for the Python bindings.

samples/
    Native SYnergy usage examples.
```

## Installation

The binding requires the custom `dpctl` fork to work. 

The expected setup for a working structure is as follows:

```text
custom dpctl fork
    -> SYnergy submodule
        -> Python bindings
```

To obtain such structure, simply go in the [`dpctl` fork](https://github.com/Gi0NZ/dpctl) and clone the repository including the submodule.

A typical `git clone` command is `git clone --branch synergy-submit-integration --recurse-submodules https://github.com/Gi0NZ/dpctl.git`

Once obtained, both SYnergy's and dpctl modules must be built. For this passage, refer to the [documentation](docs/installation.md)


## Current limitations

The current implementation is experimental and has some important limitations: 

- it requires to install the custom `dpctl` fork
- SYnergy support must be enabled at build time
- profiling requires the native SYnergy backend
- frequency scaling depends on backend, hardware and permission
- direct kernels must be defined in the native C++/SYCL layer
- OpenCL and SPIR-V runtime kernel creation depends on the backend support

For more details, refer to the [docs/limitations.md](docs/limitations.md)

## Documentation

The documentation is written with MkDocs and is available bot via [github pages](https://gi0nz.github.io/SYnergy/) and in the `docs/` folder. 

The main pages are: 

- [Installation](docs/installation.md): build requirements and setup instructions;
- [Quickstart](docs/quickstart.md): minimal examples for usign python API;
- [Direct Kernel](docs/direct_kernel.md): direct native kernel creation and submission pipeline
- [Device and Kernel Profiling](docs/device_profiling.md): profiling exposed by `SYnergyQueue`
- [Frequency Scaling](docs/frequency_scaling.md):frequency scaling and manipulation through `SYnergyDevice`
- [Limitations](docs/limitations.md): current limitations and design constraints
