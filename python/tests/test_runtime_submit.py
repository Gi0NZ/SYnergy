import ctypes
import numpy as np
import dpctl.memory as dpm
import dpctl.program as dpctl_program
import bindings._synergy_submit as synergy_submit

from pathlib import Path
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


n = 8192 * 8192

def test_opencl_cpu():
    
    print("\n \n TESTING OPENCL \n \n ")
    dev = SYnergyDevice("opencl:cpu")
    q = SYnergyQueue(dev, execution_backend="dpctl")

    print("Device:", dev)
    print("Queue device:", q.sycl_device.name)

    a_host = np.full(n, 2.0, dtype=np.float32)
    b_host = np.full(n, 3.0, dtype=np.float32)
    c_host = np.zeros(n, dtype=np.float32)

    a_dev = dpm.MemoryUSMShared(a_host.nbytes, queue=q)
    b_dev = dpm.MemoryUSMShared(b_host.nbytes, queue=q)
    c_dev = dpm.MemoryUSMShared(c_host.nbytes, queue=q)

    q.memcpy(a_dev, a_host, a_host.nbytes)
    q.memcpy(b_dev, b_host, b_host.nbytes)
    q.wait()

    event = q.submit_opencl_source(
        source=OPENCL_SRC,
        kernel_name="vector_add",
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



def test_direct_vecadd_gpu():

    print("\n \n TESTING DIRECT VECADD \n \n ")
    dev = SYnergyDevice("cuda:gpu:0")
    q = SYnergyQueue(dev)


    print("Device: ", dev.name)
    print("Queue: ", q.synergy_backend_name)

    repeat = 128
    local_size = 256

    kernel = synergy_submit.create_vecadd_kernel(q._adapter)

    print("Kernel:", kernel.get_function_name())


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
            ctypes.c_uint32(repeat),
        ],
        gS=[n],
        lS=[local_size],
        use_device_profiling=True,
        use_kernel_profiling=True,
    )

    event.wait()

    q.memcpy(c_host, c_dev, c_host.nbytes)
    q.wait()

    print("First values:", c_host[:10])


def test_spirv_kernel():
    
    print("\n\n TEST SPIRV \n\n")
    dev = SYnergyDevice("opencl:cpu:0")
    q = SYnergyQueue(dev, execution_backend="dpctl")

    a_host = np.full(n, 2.0, dtype=np.float32)
    b_host = np.full(n, 3.0, dtype=np.float32)
    c_host = np.zeros(n, dtype=np.float32)

    a_dev = dpm.MemoryUSMShared(a_host.nbytes, queue=q)
    b_dev = dpm.MemoryUSMShared(b_host.nbytes, queue=q)
    c_dev = dpm.MemoryUSMShared(c_host.nbytes, queue=q)

    q.memcpy(a_dev, a_host, a_host.nbytes)
    q.memcpy(b_dev, b_host, b_host.nbytes)
    c_dev.memset(0)

    here = Path(__file__).resolve().parent
    spirv_path = here / "spirv_kernels" / "vector_add.spv"
    spirv_bytes = spirv_path.read_bytes()

    event = q.submit_spirv(
        spirv=spirv_path,
        kernel_name="vector_add",
        args=[
            a_dev,
            b_dev,
            c_dev,
            ctypes.c_uint32(n),
            ctypes.c_uint32(128),
        ],
        gS=[n],
        lS=[256],
    )
    event.wait()

    q.memcpy(c_host, c_dev, c_host.nbytes)

    print("First values: ", c_host[:10])






if __name__ == "__main__":
    test_opencl_cpu()
    test_direct_vecadd_gpu()
    test_spirv_kernel()