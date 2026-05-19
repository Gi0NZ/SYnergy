import ctypes

import numpy as np
import dpctl
import dpctl.memory as dpm
import dpctl.program as dpctl_program


n = 4 * 1024 * 1024
repeat = 256
local_size = 256

# Usa OpenCL, non cuda, per create_program_from_source
DEVICE_SELECTOR = "opencl:cpu:0"
# oppure, se disponibile:
# DEVICE_SELECTOR = "opencl:gpu:0"


VECTOR_ADD_SOURCE = r"""
__kernel void vector_add(
    __global const float* a,
    __global const float* b,
    __global float* c,
    unsigned int n,
    unsigned int repeat
) {
    unsigned int i = get_global_id(0);

    if (i < n) {
        float value = a[i] + b[i];

        for (unsigned int r = 0; r < repeat; ++r) {
            value = value + 0.0f;
        }

        c[i] = value;
    }
}
"""


def main():
    q = dpctl.SyclQueue(
        DEVICE_SELECTOR
    )

    print("Queue:", q)
    print("Device:", q.sycl_device.name)
    print("Backend:", q.sycl_device.backend)

    a_host = np.ones(n, dtype=np.float32)
    b_host = np.full(n, 2.0, dtype=np.float32)
    c_host = np.empty_like(a_host)

    a_dev = dpm.MemoryUSMShared(a_host.nbytes, queue=q)
    b_dev = dpm.MemoryUSMShared(b_host.nbytes, queue=q)
    c_dev = dpm.MemoryUSMShared(c_host.nbytes, queue=q)

    q.memcpy(a_dev, a_host, a_host.nbytes)
    q.memcpy(b_dev, b_host, b_host.nbytes)
    c_dev.memset(0)

    program = dpctl_program.create_program_from_source(
        q,
        VECTOR_ADD_SOURCE,
    )

    kernel = program.get_sycl_kernel("vector_add")

    event = q.submit(
        kernel,
        args=[
            a_dev,
            b_dev,
            c_dev,
            ctypes.c_uint32(n),
            ctypes.c_uint32(repeat),
        ],
        gS=[n],
        lS=[local_size],
    )

    event.wait()

    q.memcpy(c_host, c_dev, c_host.nbytes)

    print("10 valori:", c_host[:10])
if __name__ == "__main__":
    raise SystemExit(main())