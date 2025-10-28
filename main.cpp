#include "ezcl.hpp"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    cl_uint numPlatforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        cerr << "No OpenCL platforms found.\n";
        return 1;
    }

    vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    int platformIndex = 0; // or user input
    cl_platform_id platform = platforms[platformIndex];

    cl_uint numDevices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        cerr << "No devices found for platform " << platformIndex << ".\n";
        return 1;
    }

    vector<cl_device_id> devices(numDevices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);
    int deviceIndex = 0; // or user input
    cl_device_id device = devices[deviceIndex];

    { // ezcl
        constexpr size_t s = 10;
        using type = int32_t;

        vector<type> a(s);
        vector<type> b(s);
        vector<type> c(s);

        for (size_t i = 0; i < s; i++) {
            a[i] = i;
            b[i] = s - i;
        }

        ezcl::Device dev(platform, device);

        ezcl::Array clA(dev, ezcl::READ_ONLY, a);
        ezcl::Array clB(dev, ezcl::READ_ONLY, b);
        ezcl::Array clC(dev, ezcl::WRITE_ONLY, c);

        dev.add(clA, clB, clC);

        clC.read(c);

        for (int i = 0; i < s; i++) {
            cout << c[i] << '\n';
        }
    } // ezcl

    return 0;
}