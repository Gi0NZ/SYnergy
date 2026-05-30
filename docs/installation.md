# Installation

This page describes how to build and use SYnergy Python bindings inside the custom `dpctl` integration.

The main needed components to utilize the binding are:

- a SYCL compiler of your choice
- `dpctl` from the [custom fork](https://github.com/Gi0NZ/dpctl/tree/synergy-submit-integration)
- the SYnergy submodule downloaded and built


## Requirements

The current implementation requires:

- a C++-compatible compiler
- Intel oneAPI DPC++ / SYCL compiler
- pybind11
- Cython
- CMake
- a supported SYCL backend (cuda, opencl, level-zero...)

Since the binding has been mainly tested on NVIDIA, for CUDA-based execution the system must also provide:

- a compatible NVIDIA driver
- CUDA support in the DPC++ toolchain
- a GPU visible through `dpctl`

## Installation and building process

In order to be able to use the API you first have to obtain the repository. 
To do so:

1. Navigate to the [repository](https://github.com/Gi0NZ/dpctl/tree/synergy-submit-integration) 
2. Copy the clone link 
3. Open a git-enabled terminal and execute `git clone --branch synergy-submit-integration --recurse-submodules https://github.com/Gi0NZ/dpctl.git`. This command will allow you to clone the repo including the SYnergy submodule

Note: from now on it is suggested to use a dedicated environment, created via pip, conda etc.

4. Move inside the SYnergy/build folder - if not present, simply create it 
5. Build the submodule, based on the backend, with:
```text
cd SYnergy
mkdir build && cd build/


cmake .. -DSYNERGY_BUILD_SAMPLES=ON -DSYNERGY_SYCL_IMPL=[OpenSYCL | DPC++] -DSYNERGY_CUDA_SUPPORT=ON

cmake .. -DSYNERGY_BUILD_SAMPLES=ON -DSYNERGY_SYCL_IMPL=[OpenSYCL | DPC++] -DSYNERGY_ROCM_SUPPORT=ON

cmake .. -DSYNERGY_BUILD_SAMPLES=ON -DSYNERGY_SYCL_IMPL=[OpenSYCL | DPC++] -DSYNERGY_LZ_SUPPORT=ON

cmake .. -DSYNERGY_BUILD_SAMPLES=ON -DSYNERGY_SYCL_IMPL=DPC++ -DSYNERGY_GEOPM_SUPPORT=ON -DSYNERGY_DEVICE_PROFILING=ON -DSYNERGY_HOST_PROFILING=ON -DSYNERGY_KERNEL_PROFILING=ON

make -j 


Note that this version of the command is for the very basic building process. You will have to then enable device and kernel profiling via `ccmake ..`

Otherwise, you can use a more complex one, as the following:

ONEAPI_DEVICE_SELECTOR=cuda:* LD_LIBRARY_PATH=your_lib_path cmake -S . -B build \
-DCMAKE_C_COMPILER=your_compiler_path \
-DCMAKE_CXX_COMPILER=your_DPC++/clang++ path \
-DPython3_EXECUTABLE=your_python_path
-DDPCTL_ROOT=your_dpctl_path
-DSYNERGY_BUILD_SAMPLES=ON \
-DSYNERGY_SYCL_IMPL=DPC++ \
-DSYNERGY_CUDA_SUPPORT=ON \
-DSYNERGY_SAMPLES_CUDA_ARCH=your_device_smcode (if NVIDIA) \
-DSYNERGY_DEVICE_PROFILING=ON \
-DSYNERGY_KERNEL_PROFILING=ON \
-DSYNERGY_USE_PROFILING_ENERGY=ON
```
6. Once the building process succeds, execute `make -j`

We have now built the SYnergy submodule, but we still need to build the dpctl one. 

1. Go into the main `dpctl/` folder
2. Execute the included build script via `python scripts/build_locally.py`
3. For further info on the dpctl build refer to the [dpctl documentation](https://intelpython.github.io/dpctl/latest/index.html#)

