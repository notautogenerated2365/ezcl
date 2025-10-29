#include <iostream>
#include <vector>

#include "ezcl.hpp"

int main() {
    std::vector<ezcl::PlatformId> plats = ezcl::getPlatforms();
    size_t maxCompUnits = 0;
    size_t platIndex;
    size_t devIndex;
    
    // display all OpenCL platforms and devices, pick the one with the most reported compute units
    for (size_t i = 0; i < plats.size(); i++) {
        const ezcl::PlatformId p = plats[i];

        std::cout << "Platform [" << i << "]: " << p.name() << '\n';
        std::cout << "  Vendor:  " << p.vendor() << "\n";
        std::cout << "  Version: " << p.version() << "\n";

        const std::vector<ezcl::DeviceId>& devices = p.getDevices();

        for (size_t j = 0; j < devices.size(); j++) {
            const auto& d = devices[j];

            std::cout << "    Device [" << j << "]: " << d.name() << " (" << d.typeString() << ")\n";
            std::cout << "      Compute Units: " << d.computeUnits() << "\n";
            std::cout << "      Memory: " << d.memSize() / (1024 * 1024) << " MB\n";

            if (d.computeUnits() > maxCompUnits) {
                maxCompUnits = d.computeUnits();
                platIndex = i;
                devIndex = j;
            }
        }

        std::cout << '\n';
    }

    // pick the one with the most compute units
    ezcl::PlatformId platform = plats[platIndex];
    ezcl::DeviceId device = platform.getDevices()[devIndex];

    // set to any size and compatible type
    constexpr size_t s = 100;
    using type = int32_t;

    // can be std::vector/array, or C-style array
    std::vector<type> a(s);
    std::vector<type> b(s);
    std::vector<type> c(s);

    // initialize two operand vectors with values
    for (size_t i = 0; i < s; i++) {
        a[i] = i;
        b[i] = s - i;
    }

    // initialize OpenCL device with platform and device we selected earlier
    ezcl::Device dev(platform, device);

    // load vectors into the device
    ezcl::Array clA(dev, ezcl::READ_ONLY, a);
    ezcl::Array clB(dev, ezcl::READ_ONLY, b);
    ezcl::Array clC(dev, ezcl::WRITE_ONLY, c);

    // perform the operation
    dev.add(clA, clB, clC);

    // read back the result into vector c
    clC.read(c);

    // display results
    for (size_t i = 0; i < s; i++) {
        std::cout << c[i] << '\n';
    }

    return 0;
}