# SYnergy Python Binding

This project provides a Python interface that allow the user to utilize selected SYnergy features through a dpctl/SYCL pipeline.

The main goal of this binding layer is to make accessible the eneregy-aware SYCL implementation of SYnergy while keeping a simple and familiar interface from the Python's dpctl implementation. 

The library introduces 2 main abstractions:

- `SYnergyDevice`, responsible for device-level operations such as frequency queries and scaling, power and energy usage.
- `SYnergyDevice`, responsible for kernel submission, queue-level execution and optional energy profiling

This design intentionally separates **device management** from **kernel execution**.


## Project Goal 

The project aims to connect three layers:

```text
Python
    -> dpctl-compatible facade
        -> Cython / pybind11 native bridge
            -> SYnergy / SYCL backend
```

This structure allows Python code to interact with SYCL devices and queues, submit kernel and collect profiling informations while delegating the low-level execution to the native SYnergy and SYCL infrastructure.

The binding is designed as a thin layer, not re-implementing SYnergy's logic, but exposing selected native capabilities via a more Python-friendly way. 

# Main Features

The current implementation supports:
 
- creation of SYnergy-aware devices from `dpctl` selectors;
- querying device name, backend, CPU/GPU type and SYnergy support;
- querying supported core and uncore frequencies;
- reading current core and uncore frequencies;
- setting core, uncore, or combined frequencies;
- creating a `SYnergyQueue` on top of a `dpctl.SyclQueue`;
- submitting existing `dpctl.program.SyclKernel` objects;
- collecting optional device-level and kernel-level profiling informations;
- creating and submitting kernels from OpenCL source through `dpctl`
- creating and submitting kernels from SPIRV-V binaries through `dpctl`
- creating and submitting direct kernels from a, ad-hoc implemented solution
- using native precompiled test kernels to validate the Synergy submit path

# Design Overview
The design Python layer is organized around two public classes.

## SYnergyDevice
`SYnergyDevice` represents a SYnergy-compatibile SYCL device. It wraps a `dpctl.SyclDevice` and, when possibile, connects it to the SYnergy backend. 

It is responsible for:

- device information;
- backend detection;
- frequency management;
- power and energy readings;

Also, frequency scaling is handled at this level.

## SYnergyQueue

`SYnergyQueue` extends `dpctl.SyclQueue` and provides a SYnergy-aware execution path. 

It is responsible for:

- kernel submission;
- optional energy profiling;
- storing the last submitted event;
- storing profiling metadata;

# Native Bridge

The native bridge is implemented via a combination of `pybind11` and Cython.

The overall structure is as follows:

```text
SYnergyDevice
    -> bindings._synergy_native.SYnergy_Device_Adapter
        -> synergy::device


SYnergyQueue
    -> bindings._synergy_native.SYnergy_Queue_Adapter
        -> bindings._synergy_submit
            -> dpctl/SYnergy submit interface
                -> synergy::queue
```

All these implementation details are hidden behind the Python API.
Apart from the use of direct kernels, the user is intended to mainly work with `SYnergyDevice` and `SYnergyQueue`.

# Documentation Structure
Use the following sections to naviate the documentation:

- [Installation](installation.md): build requirements and setup instructions.
- [QuickStart](quickstart.md): minimal examples to use the API.
- [Device Management](frequency_scaling.md): frequency scaling and device-level operations.
- [Queue and Profiling](device_profiling.md): kernel submission and profiling workflow.
- [API Reference](api.md): automatically generated API documentation from Python docstrings.
- [Limitations](limitations.md): current constraints and known design choices

# Current Status

The binding currently focuses on exposing a small but meaningful subset of SYnergy functionality to Python, including:

- device frequency management;
- queue-based kernel submission;
- energy profiling;
- runtime OpenCL/SPIR-V kernel creation through `dpctl`

