# Direct Kernel 

This page describes the direct kernel execution path used by the SYnergy Python binding.

While with OpenCL and SPIR-V one can obtain a `dpctl.program.SyclKernel` via the `create_program_from_source()` and `create_program_from_spirv` functions, obtaining such reference for a direct kernel is not quite as easy. 

Here, the kernel needs to be compiled into the native C++/SYCL extension and then be exposed through a small factory function. 

The direct kernel creation path follows this pipeline:

```text
C++ SYCL functor / kernel
    -> native factory function
        -> DPCTLSyclKernelRef
            -> Cython bridge
                -> dpctl.program.SyclKernel
                    -> SYnergyQueue.submit(...)
```

In simpler words, here the kernel is not compiled at runtime, instead it is defined in the native layer, compiled with the extension module, wrapped as `dpctl.program.SyclKernel` and then submitted from from Python. 

## Native kernel definition

The first step concern defining the kernel in C++/SYCL. You can find examples of this in `SYnergy/binding/kernels`.

The conceptual definition can be shown as follows:

```cpp
struct ExampleKernel {
    float* a;
    float* b;
    float* c;
    std::uint32_t n;

    void operator()(sycl::id<1> idx) const {
        std::uint32_t i = static_cast<std::uint32_t>(idx[0]);

        if (i < n) {
            c[i] = a[i] + b[i];
        }
    }
};
```

## Native factory function

Once the kernel is defined, the native layer must expose a factory function, whose purpose is to retrieve the compiled `sycl::kernel` object and wrap it into a `DPCTLSyclKernelRef`.

```text
native SYCL kernel
    -> sycl::kernel
        -> DPCTLSyclKernelRef
```

A factory has this shape:

```cpp
extern "C" DPCTLSyclKernelRef SYnergyTest_CreateExampleKernel(
    std::uintptr_t AdapterHandle
);
```

It can be observed in the `SYnergy/binding/synergy_test_kernels.cpp` and `SYnergy/binding/synergy_test_kernels.hpp` files. 

The `AdapterHandle` is used to recover the SYnergy queue adapter, through which the native code can accesso the underlying SYCL context and device. 

The factory handles wrapping the resulting `sycl::kernel` into a `DPCTLSyclKernelRef`, the C-level representation used by dpctl. 

## Cython bridge

The factory is then declared inside the Cython bridge, which behaves as the connector between the native C++ factory and Python. 

A simplified Cython function has this shape:

```cython
cpdef create_example_kernel(object adapter):
    cdef uintptr_t adapter_handle
    cdef DPCTLSyclKernelRef KRef

    adapter_handle = adapter._native_handle()
    KRef = SYnergyTest_CreateExampleKernel(adapter_handle)

    if KRef == NULL:
        raise RuntimeError("Unable to create native SYnergy kernel.")

    return SyclKernel._create(KRef, "SYnergyExampleKernel")
```

The returned object is a regular dpctl.program.SyclKernel. This is important because the Python SYnergyQueue.submit(...) method expects an existing dpctl.program.SyclKernel.

To utilize this pipeline in Python the sequence is as follows:

```python
...
import bindings._synergy_submit as synergy_submit

from bindings import SYnergyQueue
...

queue = SYnergyQueue("cuda:gpu:0", execution_backend="synergy")

...

kernel = synergy_submit.create_example_kernel(queue._adapter) 

event = queue.submit(
    kernel=kernel
    ...
)

event.wait()
```


# Important note

The direct-kernel path is not meant to compile arbitrary python defined kernels at runtime. 

Every new direct kernel must be:

1. defined in the native C++/SYCL layer
2. compiled with the extension module
3. exposed through a native factory function
4. declared in the Cython bridge
5. imported and submitted from Python

Though much more complicated, these kernel have been implemented per design choice, since not every backend is able to support OpenCL/SPIR-V kernels. 