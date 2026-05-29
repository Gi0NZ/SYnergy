"""
Internal Cython bridge for SYnergy kernel submission.

This module connects the Python ``SYnergyQueue`` facade with the native
DPCTL/SYnergy submit interface. It is responsible for translating Python-side
kernel arguments, execution ranges, dependency events, and profiling flags into
the native structures expected by the C++ backend.

This module is considered internal. End users should normally interact with
``SYnergyQueue.submit()``, ``SYnergyQueue.submit_spirv()``, or
``SYnergyQueue.submit_opencl_source()`` instead of importing this module
directly.
"""

# distutils: language = c++
# cython: language_level=3

from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t
from libc.stddef cimport size_t

from dpctl._backend cimport (
    DPCTLSyclEventRef,
    DPCTLSyclKernelRef,
    _arg_data_type,
)

from dpctl._sycl_queue cimport SyclQueue
from dpctl._sycl_event cimport SyclEvent
from dpctl.program._program cimport SyclKernel


cdef extern from "syclinterface/dpctl_sycl_synergy_queue_interface.h":
    DPCTLSyclEventRef DPCTLQueue_SubmitRangeSYnergy(
        uintptr_t AdapterHandle,
        const DPCTLSyclKernelRef KRef,
        void** Args,
        const _arg_data_type* ArgTypes,
        size_t NArgs,
        const size_t Range[3],
        size_t NDims,
        const DPCTLSyclEventRef* DepEvents,
        size_t NDepEvents,
        unsigned int UncoreFrequency,
        unsigned int CoreFrequency,
        int UseFrequencyScaling,
    )

    DPCTLSyclEventRef DPCTLQueue_SubmitNDRangeSYnergy(
        uintptr_t AdapterHandle,
        const DPCTLSyclKernelRef KRef,
        void** Args,
        const _arg_data_type* ArgTypes,
        size_t NArgs,
        const size_t gRange[3],
        const size_t lRange[3],
        size_t NDims,
        const DPCTLSyclEventRef* DepEvents,
        size_t NDepEvents,
        unsigned int UncoreFrequency,
        unsigned int CoreFrequency,
        int UseFrequencyScaling
    )
cdef extern from "synergy_test_kernels.hpp":
    DPCTLSyclKernelRef SYnergyTest_CreateVecAddKernel(
        uintptr_t AdapterHandle
    )
    
    DPCTLSyclKernelRef SYnergyTest_CreateVecprodKernel(
        uintptr_t AdapterHandle
    )


cpdef submit(
    SyclQueue queue,
    object adapter,
    SyclKernel kernel,
    list args,
    list gS,
    object lS,
    list dEvents,
    bint use_device_profiling,
    bint use_kernel_profiling,
    ):
    """
    Submit a kernel through the native SYnergy backend.

    This function is the internal implementation used by
    ``SYnergyQueue.submit`` when the queue is configured with
    ``execution_backend="synergy"``. It converts Python-level objects into the
    native DPCTL/SYCL references required by the C++ submit interface.

    Parameters
    ----------
    queue : dpctl.SyclQueue
        Python queue object associated with the submission.
    adapter : object
        Native SYnergy queue adapter associated with ``queue``.
    kernel : dpctl.program.SyclKernel
        Kernel to submit.
    args : list
        Kernel arguments.
    gS : list[int]
        Global execution range.
    lS : list[int] or None
        Local execution range. If None, a range submit is used; otherwise an
        NDRange submit is used.
    dEvents : list
        Dependency events.
    use_device_profiling : bool
        Enable device-level energy profiling.
    use_kernel_profiling : bool
        Enable kernel-level energy profiling.

    Returns
    -------
    tuple
        Pair ``(event, profile)`` where ``event`` is a ``dpctl.SyclEvent`` and
        ``profile`` is a dictionary containing profiling metadata.

    Notes
    -----
    This function does not perform frequency scaling. Device frequencies are
    managed separately through ``SYnergyDevice``.

    The native C interface still accepts frequency-scaling parameters for
    compatibility with the underlying SYnergy submit functions, but this bridge
    always forwards ``0, 0, 0`` so that Python-level frequency control remains
    separated from kernel submission.
    """

    cdef void** kargs = NULL
    cdef _arg_data_type* kargty = NULL
    cdef DPCTLSyclEventRef* depEvents = NULL
    cdef DPCTLSyclEventRef Eref = NULL

    cdef size_t gRange[3]
    cdef size_t lRange[3]

    cdef size_t nArgs = len(args)
    cdef size_t nGS = len(gS)
    cdef size_t nLS = 0
    cdef size_t nDE = 0

    cdef int ret = 0
    cdef uintptr_t adapter_handle
    cdef SyclEvent event

    cdef object device_energy_before = None
    cdef object device_energy_after = None
    cdef object device_energy_delta = None
    cdef object kernel_energy = None

    if dEvents is not None:
        nDE = len(dEvents)

    if lS is not None:
        nLS = len(lS)

    adapter_handle = <uintptr_t>adapter._native_handle()

    if use_device_profiling:
        device_energy_before = adapter.device_energy_consumption()

    try:
        # ------------------------------------------------------------
        # 1. Allocate kernel arguments
        # ------------------------------------------------------------
        if nArgs > 0:
            kargs = <void**>malloc(nArgs * sizeof(void*))
            if kargs == NULL:
                raise MemoryError()

            kargty = <_arg_data_type*>malloc(nArgs * sizeof(_arg_data_type))
            if kargty == NULL:
                raise MemoryError()

            ret = queue._populate_args(args, kargs, kargty)
            if ret == -1:
                raise TypeError("Unsupported type for a kernel argument")

        # ------------------------------------------------------------
        # 2. Prepare dependency events
        # ------------------------------------------------------------
        if nDE > 0:
            depEvents = <DPCTLSyclEventRef*>malloc(
                nDE * sizeof(DPCTLSyclEventRef)
            )
            if depEvents == NULL:
                raise MemoryError()

            for idx, de in enumerate(dEvents):
                if isinstance(de, SyclEvent):
                    depEvents[idx] = (<SyclEvent>de).get_event_ref()
                else:
                    raise TypeError(
                        "dEvents must be a sequence of dpctl.SyclEvent"
                    )

        # ------------------------------------------------------------
        # 3. Prepare global execution range
        # ------------------------------------------------------------
        ret = queue._populate_range(gRange, gS, nGS)
        if ret == -1:
            raise ValueError(
                "Global range must have 1, 2 or 3 dimensions."
            )

        # ------------------------------------------------------------
        # 4. Submit Range or NDRange via SYnergy
        # ------------------------------------------------------------
        if lS is None:
            Eref = DPCTLQueue_SubmitRangeSYnergy(
                adapter_handle,
                kernel.get_kernel_ref(),
                kargs,
                kargty,
                nArgs,
                gRange,
                nGS,
                depEvents,
                nDE,
                0,
                0,
                0,
            )
        else:
            ret = queue._populate_range(lRange, <list>lS, nLS)
            if ret == -1:
                raise ValueError(
                    "Local range must have 1, 2 or 3 dimensions."
                )

            if nGS != nLS:
                raise ValueError(
                    "Local and global ranges must have the same dimensions."
                )

            for i in range(nGS):
                if lRange[i] == 0 or gRange[i] % lRange[i] != 0:
                    raise ValueError(
                        "Each global range dimension must be divisible "
                        "by the corresponding local range dimension."
                    )

            Eref = DPCTLQueue_SubmitNDRangeSYnergy(
                adapter_handle,
                kernel.get_kernel_ref(),
                kargs,
                kargty,
                nArgs,
                gRange,
                lRange,
                nGS,
                depEvents,
                nDE,
                0,
                0,
                0,
            )

        if Eref == NULL:
            raise RuntimeError("SYnergy kernel submission failed.")

        # ------------------------------------------------------------
        # 5. Wrap the native event as dpctl.SyclEvent and wait
        # ------------------------------------------------------------
        event = SyclEvent._create(Eref)
        event.wait()

        # ------------------------------------------------------------
        # 6. Optional profile informations
        # ------------------------------------------------------------
        if use_device_profiling:
            device_energy_after = adapter.device_energy_consumption()
            device_energy_delta = device_energy_after - device_energy_before

        if use_kernel_profiling:
            kernel_energy = adapter.kernel_energy_consumption(event)

        profile = {
            "use_device_profiling": bool(use_device_profiling),
            "use_kernel_profiling": bool(use_kernel_profiling),
            "device_energy_before": device_energy_before,
            "device_energy_after": device_energy_after,
            "device_energy_delta": device_energy_delta,
            "kernel_energy": kernel_energy,
        }

        return event, profile

    finally:
        if kargs != NULL:
            free(kargs)

        if kargty != NULL:
            free(kargty)

        if depEvents != NULL:
            free(depEvents)


cpdef create_vecadd_kernel(object adapter):
    """
    Create the native test vector-add kernel.

    This helper is used by tests and examples to obtain a precompiled
    ``dpctl.program.SyclKernel`` backed by the native SYnergy test kernel.

    Parameters
    ----------
    adapter : object
        Native SYnergy queue adapter used to recover the underlying SYCL
        context and device.

    Returns
    -------
    dpctl.program.SyclKernel
        Wrapped vector-add kernel.
    """
    cdef uintptr_t adapter_handle
    cdef DPCTLSyclKernelRef KRef

    adapter_handle = <uintptr_t>adapter._native_handle()

    KRef = SYnergyTest_CreateVecAddKernel(adapter_handle)

    if KRef == NULL:
        raise RuntimeError("Unable to create native SYnergy vector-add kernel.")

    return SyclKernel._create(KRef, "SYnergyVecAddKernel")


cpdef create_vecprod_kernel(object adapter):
    """
    Create the native vector-product test kernel.

    This helper returns a precompiled ``dpctl.program.SyclKernel`` backed by
    the native SYnergy vector-product test kernel. It is mainly intended for
    internal tests and benchmarking experiments that require a simple native
    GPU workload.

    Parameters
    ----------
    adapter : object
        Native SYnergy queue adapter used to recover the underlying SYCL
        context and device.

    Returns
    -------
    dpctl.program.SyclKernel
        Wrapped vector-product kernel.

    Raises
    ------
    RuntimeError
        If the native backend cannot create or wrap the vector-product kernel.
    """
    cdef uintptr_t adapter_handle
    cdef DPCTLSyclKernelRef KRef

    adapter_handle = <uintptr_t>adapter._native_handle()

    KRef = SYnergyTest_CreateVecprodKernel(adapter_handle)

    if KRef == NULL:
        raise RuntimeError("Unable to crete vecprod kernel")

    return SyclKernel._create(KRef, "SYnergyVecProdKernel")

