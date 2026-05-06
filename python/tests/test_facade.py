from bindings import SYnergyQueue


def main():
    with SYnergyQueue("cuda:gpu:0") as q:
        print("Device:", q.device_name)
        print("Backend:", q.backend_name)
        print("Device energy:", q.device_energy())
        caps = q.capabilities();
        print("Capabilities: ")
        for key, value in caps.as_dict().items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()