import numpy as np
import dpctl
import dpctl.memory as dpm
import ctypes

import bindings._synergy_submit as synergy_submit

from bindings import SYnergyQueue


N = 4 * 8192 * 8192
REPEAT = 256
LOCAL_SIZE = 256
DEVICE_SELECTOR = "cuda:gpu:0"

def allocate_usm_vectors(q: SYnergyQueue, n: int):
    """
    Creates and allocates 3 vectors
    
    Data is allocated as USM Shared so they can be sent as args to SYCL Kernel 
    """

    a_host = np.full(n, 2.0, dtype=np.float32)
    b_host = np.full(n, 2.0, dtype=np.float32)
    c_host = np.empty_like (a_host)

    a_dev = dpm.MemoryUSMShared(a_host.nbytes, queue= q)
    b_dev = dpm.MemoryUSMShared(b_host.nbytes, queue= q)
    c_dev = dpm.MemoryUSMShared(c_host.nbytes, queue= q)
    
    q.memcpy(a_dev, a_host, a_host.nbytes)
    q.memcpy(b_dev, b_host, b_host.nbytes)
    c_dev.memset(0)

    return a_dev, b_dev, c_dev, c_host


def print_results(q: SYnergyQueue, c_dev, c_host) -> bool:
    q.memcpy(c_host, c_dev, c_host.nbytes)

    print("Vettore risultante:", c_host[:10])



def main() -> int:
    print("Creating Queue")

    q = SYnergyQueue(
        DEVICE_SELECTOR,
        execution_backend="synergy"
        )
    
    print("Used device: ", q.synergy_device_name)

    if not hasattr(synergy_submit, "create_vecprod_kernel"):
        raise RuntimeError(
            "bindings._synergy_submit does not expose create_vecprod_kernel()\n"
            "Make sure the name is correct and that _synergy_submit has been compiled correctly"
        )
    
    #Creazione SyclKernel
    kernel = synergy_submit.create_vecprod_kernel(q._adapter)

    try:
        print("Kernel name:", kernel.get_function_name())
    except Exception as exc:
        print("Kernel name unavailable", repr(exc))

    a_dev, b_dev, c_dev, c_host = allocate_usm_vectors(q, N)

    print()

    event = q.submit(
        kernel,
        args=[
            a_dev,
            b_dev,
            c_dev,
            ctypes.c_uint32(N),
            ctypes.c_uint32(REPEAT),
        ],
        gS=[N],
        lS=[LOCAL_SIZE],
        use_device_profiling=True,
        use_kernel_profiling=True,
    )
    event.wait()

    print("=====Results=====")

    try:
        print("submit ns:", event.profiling_info_submit)
        print("start ns:", event.profiling_info_start)
        print("end ns:", event.profiling_info_end)
        print("duration ns:", event.profiling_info_end - event.profiling_info_start)
        print()
        print(q.last_profile)
        print()
        print_results(q, c_dev, c_host)
    except Exception as exc:
        print("Event profiling info unavailable:", repr(exc))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
