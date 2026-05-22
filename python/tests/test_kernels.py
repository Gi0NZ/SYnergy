import ctypes
import numpy as np
import dpctl.memory as dpm
import dpctl.program as dpctl_program

from bindings import SYnergyDevice, SYnergyQueue


OPENCL_SRC = r"""
__kernel void vector_add(
    __global const float* a,
    __global const float* b,
    __global float* c,
    unsigned int n
) {
    size_t i = get_global_id(0);

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
"""


def main():
    n = 1024 * 1024

    dev = SYnergyDevice("cpu")
    q = SYnergyQueue(dev, execution_backend="dpctl")

    print("Device:", dev)
    print("Queue device:", q.sycl_device.name)

    program = dpctl_program.create_program_from_source(q, OPENCL_SRC)
    kernel = program.get_sycl_kernel("vector_add")

    print("Kernel:", kernel.get_function_name())
    print("Num args:", kernel.get_num_args())

    a_host = np.full(n, 2.0, dtype=np.float32)
    b_host = np.full(n, 3.0, dtype=np.float32)
    c_host = np.zeros(n, dtype=np.float32)

    a_dev = dpm.MemoryUSMShared(a_host.nbytes, queue=q)
    b_dev = dpm.MemoryUSMShared(b_host.nbytes, queue=q)
    c_dev = dpm.MemoryUSMShared(c_host.nbytes, queue=q)

    q.memcpy(a_dev, a_host, a_host.nbytes)
    q.memcpy(b_dev, b_host, b_host.nbytes)
    q.wait()
    event = q.submit(
        kernel,
        args=[
            a_dev,
            b_dev,
            c_dev,
            ctypes.c_uint32(n),
        ],
        gS=[n],
        lS=[256],
    )

    event.wait()

    q.memcpy(c_host, c_dev, c_host.nbytes)
    q.wait()

    print("First values:", c_host[:10])
    print("Expected:", 5.0)

    assert np.allclose(c_host, 5.0)
    print("OK")


if __name__ == "__main__":
    main()