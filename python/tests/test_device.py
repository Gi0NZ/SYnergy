from bindings import SYnergyDevice, SYnergyQueue


def test_gpu_queue_from_device():
    print("\n=== GPU queue from SYnergyDevice ===")

    dev = SYnergyDevice("cuda:gpu:0")
    q = SYnergyQueue(dev, execution_backend="synergy")

    print("device:", dev)
    print("queue synergy device:", q.synergy_device_name)
    print("queue synergy backend:", q.synergy_backend_name)

    assert dev.is_synergy_supported is True
    assert q.synergy_device_name is not None


def test_gpu_queue_from_string():
    print("\n=== GPU queue from selector string ===")

    q = SYnergyQueue("cuda:gpu:0", execution_backend="synergy")

    print("queue synergy device:", q.synergy_device_name)
    print("queue synergy backend:", q.synergy_backend_name)

    assert q.synergy_device_name is not None


def test_cpu_device_creation():
    print("\n=== CPU SYnergyDevice creation ===")

    dev = SYnergyDevice("cpu")

    print("device:", dev)
    print("synergy supported:", dev.is_synergy_supported)

    assert dev.is_cpu is True
    assert dev.is_synergy_supported is False


def test_cpu_queue_dpctl_mode():
    print("\n=== CPU queue in dpctl mode ===")

    dev = SYnergyDevice("cpu")
    q = SYnergyQueue(dev, execution_backend="dpctl")

    print("device:", dev)
    print("queue device:", q.sycl_device.name)

    assert q.sycl_device.is_cpu


def test_device_frequencies():
    print("\n=== GPU supported Frequencies ===")

    dev = SYnergyDevice("cuda:gpu:0")
    print("Core Frequencies:", dev.supported_core_frequencies())
    print("Uncore Frequencies:", dev.supported_uncore_frequencies())


def main():
    test_gpu_queue_from_device()
    test_gpu_queue_from_string()
    test_cpu_device_creation()
    test_cpu_queue_dpctl_mode()
    test_device_frequencies()



if __name__ == "__main__":
    main()