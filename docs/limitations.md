# Limitations

This page summarizes the current limitations of the SYnergy Python bindings and of the custom dpctl integration.

## Custom `dpctl` fork required

The current implementation requires the custom  `dpctl` fork with SYnergy support enabled, since the standard upstream version of `dpctl` does not include the synergy submit integration.

## SYnergy backend availability

`SYnergyQueue` can operate with different execution backends. 

When the queue uses `execution_backend="synergy"`, kernel submissions are routed through the native SYnergy backend; instead when it uses `execution_backend="dpctl"` the queue follows the standard execution path. 

In the latter, basic kernel execution could still work flawlessly, but SYnergy-specific features such as profiling and frequency scaling submissions, are not available. 

## Backend-dependent support

Not all SYCL backends expose the same features. 

The current implementation can depend on backend-specific support. 

This means that a given device could be visible through `dpctl` but still not support all SYnergy operations. 

For this reason, applications should inspect the queue and device capabilities before relying on profiling or frequency-scaling features.

## Direct-kernel limitations

Direct kernels cannot be defined directly through Python.

A direct kernel must be:

1. defined in the native C++/SYCL layer;
2. compiled with the native extension;
3. exposed through a factory function;
4. wrapped as a DPCTLSyclKernelRef;
5. converted into a dpctl.program.SyclKernel through the Cython bridge;
6. submitted through SYnergyQueue.submit(...).

For the complete explanation refer to [Direct Kernel](direct_kernel.md)

This means that the direct-kernel path is suitable for predefined kernels, tests and controlled experiments, but it is not a runtime Python kernel-generation mechanism.

## OpenCL and SPIR-V limitations
Unlike direct kernels, OpenCL and SPIR-V ones can be submitted through helper methods such as `queue.submit_opencl_source(...)` and `queue.submit_spirv(...)`.

However, support for runtime kernel creation depends on the selected backend. 

For example, a backend may support queue execution but not support runtime compilation from OpenCL source. In that case, `submit_opencl_source(...)` may fail even if the selected device is visible through `dpctl`.

It is then advised to separate kernel creation and kernel execution. 

## Python package

The Python package is currently located under: `SYnergy/python/bindings/`. During development, this directory usually has to be added to `PYTHONPATH`.