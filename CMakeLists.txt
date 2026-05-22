cmake_minimum_required(VERSION 3.17)
project(SYnergy VERSION 0.1.1 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SYNERGY_CUDA_SUPPORT "Enable CUDA support" OFF)
option(SYNERGY_ROCM_SUPPORT "Enable ROCm support" OFF)
option(SYNERGY_LZ_SUPPORT "Enable Level Zero Support" OFF)
option(SYNERGY_GEOPM_SUPPORT "Enable GEOPM Support" OFF)
option(SYNERGY_USE_PROFILING_ENERGY "Enable profiling by utilizing energy counter" OFF)

set(SYNERGY_SYCL_IMPL "" CACHE STRING "Select SYCL implementation [OpenSYCL | DPC++]")
set_property(CACHE SYNERGY_SYCL_IMPL PROPERTY STRINGS "OpenSYCL" "DPC++")

if(SYNERGY_SYCL_IMPL STREQUAL "OpenSYCL")
	find_package(OpenSYCL)
	
	if(NOT OpenSYCL_FOUND)
		find_package(hipSYCL)
	endif()

	if(NOT OpenSYCL_FOUND AND NOT hipSYCL_FOUND)
		message(SEND_ERROR "OpenSYCL or hipSYCL package were not found. Specify OpenSYCL_DIR or hipSYCL_DIR.")
	endif()
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING
		"Choose the type of build, options are: Debug Release RelWithDebInfo." FORCE)
	message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified")
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

# ###################### SYnergy ######################
find_package(Threads REQUIRED)

add_library(synergy INTERFACE)

target_link_libraries(synergy INTERFACE Threads::Threads)

target_include_directories(synergy INTERFACE include)
target_include_directories(synergy INTERFACE include/profiling)

option(SYNERGY_DEVICE_PROFILING "Enable device energy consumption profiling" OFF)
option(SYNERGY_KERNEL_PROFILING "Enable kernel energy consumption profiling" OFF)

if (SYNERGY_USE_PROFILING_ENERGY)
	target_compile_definitions(synergy INTERFACE SYNERGY_USE_PROFILING_ENERGY)
endif()

if(SYNERGY_DEVICE_PROFILING)
	option(SYNERGY_HOST_PROFILING "Enable host energy consumption profiling" OFF)
	target_compile_definitions(synergy INTERFACE SYNERGY_DEVICE_PROFILING)
	if (SYNERGY_HOST_PROFILING)
		target_compile_definitions(synergy INTERFACE SYNERGY_HOST_PROFILING)
	endif()
endif()

if(SYNERGY_KERNEL_PROFILING)
	target_compile_definitions(synergy INTERFACE SYNERGY_KERNEL_PROFILING)
endif()

if (SYNERGY_GEOPM_SUPPORT)
	# if (SYNERGY_DEVICE_PROFILING)
	# 	message(WARNING "GEOPM support is not compatible with device profiling. Disabling device profiling.")
	# endif()
	target_compile_definitions(synergy INTERFACE SYNERGY_GEOPM_SUPPORT)
	target_link_libraries(synergy INTERFACE geopmd)
	target_sources(synergy INTERFACE include/vendors/geopm_wrapper.hpp)
endif()

if(SYNERGY_CUDA_SUPPORT)
	find_package(CUDAToolkit REQUIRED)

	target_compile_definitions(synergy INTERFACE "SYNERGY_CUDA_SUPPORT")
	target_link_libraries(synergy INTERFACE CUDA::nvml)
	target_sources(synergy INTERFACE include/vendors/nvml_wrapper.hpp)
endif()

if(SYNERGY_ROCM_SUPPORT)
	list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
	find_package(rocm_smi REQUIRED)

	target_compile_definitions(synergy INTERFACE SYNERGY_ROCM_SUPPORT)
	target_link_libraries(synergy INTERFACE ${ROCM_SMI_LIBRARIES})
	target_sources(synergy INTERFACE include/vendors/rsmi_wrapper.hpp)
endif()

if(SYNERGY_LZ_SUPPORT)
	find_package(LevelZero REQUIRED)

	target_compile_definitions(synergy INTERFACE SYNERGY_LZ_SUPPORT)
	target_link_libraries(synergy INTERFACE LevelZero::LevelZero)
	target_sources(synergy INTERFACE include/vendors/lz_wrapper.hpp)
endif()

# ##################### Samples ######################
option(SYNERGY_BUILD_SAMPLES "Build samples" OFF)
option(SYNERGY_BUILD_TEST "Build test" OFF)


if(SYNERGY_BUILD_SAMPLES)
	add_subdirectory(samples)
endif()

if(SYNERGY_BUILD_TEST)
	add_subdirectory(test)
endif()

# ==========================================
# SEZIONE AGGIUNTIVA: Binding Python - C++
# ==========================================

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Trova pybind11 usando il Python dell'ambiente scelto da Python3_EXECUTABLE
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED CONFIG)

# Recupera include e link flags di dpctl
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m dpctl --includes
    OUTPUT_VARIABLE DPCTL_INCLUDES_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

separate_arguments(DPCTL_INCLUDES UNIX_COMMAND "${DPCTL_INCLUDES_RAW}")

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m dpctl --library
    OUTPUT_VARIABLE DPCTL_LIBRARY_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

separate_arguments(DPCTL_LIBRARY_FLAGS UNIX_COMMAND "${DPCTL_LIBRARY_RAW}")

message(STATUS "Python3_EXECUTABLE = ${Python3_EXECUTABLE}")
message(STATUS "pybind11_DIR = ${pybind11_DIR}")
message(STATUS "DPCTL_INCLUDES = ${DPCTL_INCLUDES}")
message(STATUS "DPCTL_LIBRARY_FLAGS = ${DPCTL_LIBRARY_FLAGS}")

# Modulo nativo usato dalla facade Python:
# python/bindings/synergy_queue.py importa bindings._synergy_native
pybind11_add_module(_synergy_native
    binding/gpu_synergy_profiling_binding.cpp
)

# Header di dpctl, necessario per dpctl4pybind11.hpp
target_compile_options(_synergy_native PRIVATE
    ${DPCTL_INCLUDES}
)

# Collega la libreria SYnergy.
# Questo fa ereditare include/, include/profiling, CUDA::nvml e macro abilitate.
target_link_libraries(_synergy_native PRIVATE synergy)

# Per sicurezza, rendiamo esplicite anche sul binding le macro di profiling.
target_compile_definitions(_synergy_native PRIVATE
    SYNERGY_DEVICE_PROFILING
    SYNERGY_KERNEL_PROFILING
    SYNERGY_USE_PROFILING_ENERGY
)

# Compilazione SYCL per NVIDIA CUDA, RTX 3070 Laptop = sm_86
target_compile_options(_synergy_native PRIVATE
    -fsycl
    -fsycl-targets=nvptx64-nvidia-cuda
    "SHELL:-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_86"
)

target_link_options(_synergy_native PRIVATE
    -fsycl
    -fsycl-targets=nvptx64-nvidia-cuda
    "SHELL:-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_86"
    ${DPCTL_LIBRARY_FLAGS}
)

# Output del modulo Python compilato
set_target_properties(_synergy_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/bindings"
)

# ==========================================
# Binding Cython per submit SYnergy
# ==========================================

set(SYNERGY_SUBMIT_PYX
    "${CMAKE_CURRENT_SOURCE_DIR}/python/bindings/_synergy_submit.pyx"
)

set(SYNERGY_SUBMIT_CPP
    "${CMAKE_CURRENT_BINARY_DIR}/_synergy_submit.cpp"
)

# Trova la directory Python che contiene il package dpctl,
# così Cython può risolvere cimport dpctl._sycl_queue, ecc.
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import dpctl, pathlib; print(pathlib.Path(dpctl.__file__).parent.parent)"
    OUTPUT_VARIABLE DPCTL_PXD_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "DPCTL_PXD_ROOT = ${DPCTL_PXD_ROOT}")

add_custom_command(
    OUTPUT ${SYNERGY_SUBMIT_CPP}
    COMMAND ${Python3_EXECUTABLE} -m cython
            --cplus
            -3
            -I ${DPCTL_PXD_ROOT}
            -I ${CMAKE_CURRENT_SOURCE_DIR}/python
            -o ${SYNERGY_SUBMIT_CPP}
            ${SYNERGY_SUBMIT_PYX}
    DEPENDS ${SYNERGY_SUBMIT_PYX}
    COMMENT "Cythonizing _synergy_submit.pyx"
)

pybind11_add_module(_synergy_submit
    ${SYNERGY_SUBMIT_CPP}
	binding/synergy_test_kernels.cpp
)

target_include_directories(_synergy_submit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/binding
)

target_compile_options(_synergy_submit PRIVATE
    ${DPCTL_INCLUDES}
)

target_link_libraries(_synergy_submit PRIVATE synergy)

set(DPCTL_SYCL_INTERFACE_LIB
    "${DPCTL_ROOT}/dpctl/libDPCTLSyclInterface.so"
)

if(NOT EXISTS "${DPCTL_SYCL_INTERFACE_LIB}")
    message(FATAL_ERROR
        "libDPCTLSyclInterface.so not found at ${DPCTL_SYCL_INTERFACE_LIB}"
    )
endif()

target_link_libraries(_synergy_submit PRIVATE
    "${DPCTL_SYCL_INTERFACE_LIB}"
)

set_target_properties(_synergy_submit PROPERTIES
    BUILD_RPATH "${DPCTL_ROOT}/dpctl"
    INSTALL_RPATH "${DPCTL_ROOT}/dpctl"
)

target_link_libraries(_synergy_native PRIVATE
    "${DPCTL_SYCL_INTERFACE_LIB}"
)

set_target_properties(_synergy_native PROPERTIES
    BUILD_RPATH "${DPCTL_ROOT}/dpctl"
    INSTALL_RPATH "${DPCTL_ROOT}/dpctl"
)

target_compile_definitions(_synergy_submit PRIVATE
    SYNERGY_DEVICE_PROFILING
    SYNERGY_KERNEL_PROFILING
    SYNERGY_USE_PROFILING_ENERGY
)

target_compile_options(_synergy_submit PRIVATE
    -fsycl
    -fsycl-targets=nvptx64-nvidia-cuda
    "SHELL:-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_86"
)

target_link_options(_synergy_submit PRIVATE
    -fsycl
    -fsycl-targets=nvptx64-nvidia-cuda
    "SHELL:-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_86"
    ${DPCTL_LIBRARY_FLAGS}
)

set_target_properties(_synergy_submit PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/bindings"
)