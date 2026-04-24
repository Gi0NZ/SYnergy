
#include <iostream>
#include <nvml.h>

int main() {
    nvmlReturn_t result;

    // 1. Inizializziamo il driver NVIDIA
    result = nvmlInit();
    if (NVML_SUCCESS != result) {
        std::cout << "❌ Errore Inizializzazione NVML: " << nvmlErrorString(result) << "\n";
        return 1;
    }

    // 2. Cerchiamo la tua scheda video (Indice 0)
    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (NVML_SUCCESS != result) {
        std::cout << "❌ Impossibile agganciare la GPU: " << nvmlErrorString(result) << "\n";
        return 1;
    }

    // Stampiamo il nome per conferma
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    std::cout << "\nGPU Trovata: " << name << "\n";
    std::cout << "-------------------------------------------\n";

    // --- TEST 1: Lettura Frequenza Base (nvmlDeviceGetApplicationsClock) ---
    // Questa è la funzione esatta che ha fatto arrabbiare SYnergy alla riga 101
    unsigned int core_clock;
    result = nvmlDeviceGetApplicationsClock(device, NVML_CLOCK_GRAPHICS, &core_clock);
    std::cout << "[TEST 1] Lettura Frequenze Hardware: ";
    if (NVML_SUCCESS == result) {
        std::cout << "✅ OK (" << core_clock << " MHz)\n";
    } else {
        std::cout << "❌ BLOCCATA (" << nvmlErrorString(result) << ")\n";
    }

    // --- TEST 2: Lettura Consumo Energetico (nvmlDeviceGetPowerUsage) ---
    unsigned int power;
    result = nvmlDeviceGetPowerUsage(device, &power);
    std::cout << "[TEST 2] Lettura Potenza / Energia : ";
    if (NVML_SUCCESS == result) {
        std::cout << "✅ OK (" << (power / 1000.0) << " Watt)\n";
    } else {
        std::cout << "❌ BLOCCATA (" << nvmlErrorString(result) << ")\n";
    }

    std::cout << "-------------------------------------------\n\n";

    nvmlShutdown();
    return 0;
}