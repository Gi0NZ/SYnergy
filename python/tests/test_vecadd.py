import ctypes
import numpy as np
import dpctl
import dpctl.memory as dpm

from bindings import SYnergyDevice, SYnergyQueue
import bindings._synergy_submit as synergy_submit


def main():
    n = 8192 * 8192
    repeat = 128
    local_size = 256

    dev = SYnergyDevice("cuda:gpu:0")
    q = SYnergyQueue(dev, execution_backend="synergy")

    print("Device:", dev)
    print("Queue device:", q.sycl_device.name)

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
    print("Expected:", 5.0)

    assert np.allclose(c_host, 5.0)
    print("OK")

    print(q.last_profile)


if __name__ == "__main__":
    main()