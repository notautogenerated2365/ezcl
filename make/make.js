const fs = require("fs");
const arrays = require("./arrays.cjs");
const objects = require("./objects.cjs");

global.numType = arrays.numType;
global.opType = arrays.opType;

global.numMeta = objects.numMeta;
global.opMeta = objects.opMeta;

/*
Pseudocode:

namespace cl {
    some helper functions

    class cl {
        private:
            cl device refs/info

            some more helper functions
        
        public:
            add {
                num types
            } sub {
                num types
            } mul {
                num types
            } div {
                num types
            }
    }
}

DRY, use lots of helper functions
have public frontend and private backend functions
*/

// sample kernel:
`__kernel void <opName>_<typeName>(__global const <typeName>* a, __global const <typeName>* b, __global <typeName>* c) {
    int gid = get_global_id(0);
    c[gid] = a[gid] <opOperator> b[gid];
}`;

function make(sourcePath) {
    let source = "";

    source += `#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <stdexcept>

namespace ezcl {
    inline const char* makeKernelFunction(const char* name, const char* typeName, const char opOperator) {
        std::ostringstream function;

        function
            << "__kernel void " << name << "(__global const " << typeName << "* a, __global const " << typeName << "* b, __global " << typeName << "* c, const ulong s) {"
            << "\\n    int gid = get_global_id(0);"
            << "\\n    if (gid < s) c[gid] = a[gid] " << opOperator << " b[gid];"
            << "\\n}"
        ;
        
        return function.str().c_str();
    }

    inline void checkErr(cl_int err, const char* name) {
        if (err != CL_SUCCESS) {
            throw std::runtime_error(std::string("Error: ") + std::string(name) + std::string(" (") + std::to_string(err) + std::string(")\\n"));
        }
    }

    enum AccessType : int {
        READ_WRITE = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        READ_ONLY = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        WRITE_ONLY = CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
    };

    // required for Array::Array(Device& dev, AccessType acc, const std::vector<T>& dat)
    class Device;

    template <typename T>
    class Array {
        private:
            Device& device;
            cl_mem data;
            AccessType access;
            size_t _size;

        public:
            Array() = delete;
            Array(const Array&) = delete;

            // has to be defined after Device class definition
            Array(Device& dev, AccessType acc, const std::vector<T>& dat);
            template <size_t S>
            Array(Device& dev, AccessType acc, const std::array<T, S>& dat);
            Array(Device& dev, AccessType acc, const T* dat, const size_t s);
            Array(Array&& other) : device(other.device), data(other.data), access(other.access), _size(other._size) {
                other.data = nullptr;
                other._size = 0;
            }
            
            Device& getDevice() {return device;}
            cl_mem& getMem() {return data;}
            AccessType getAccessType() const {return access;}
            size_t getSize() const {return _size;}
            size_t size() const {return _size;}

            // has to be defined after Device class definition
            void read(std::vector<T>& v);
            template <size_t S>
            void read(std::array<T, S>& a);
            void read(T* dat, const size_t s);
            
            Array& operator=(const Array&) = delete;
            Array& operator=(Array&& other) {
                if (this != &other) {
                    if (data) clReleaseMemObject(data);

                    data = other.data;
                    access = other.access;
                    _size = other._size;
                    other.data = nullptr;
                    other._size = 0;
                }
                
                return *this;
            }

            ~Array() {
                if (data) {
                    clReleaseMemObject(data);
                    data = nullptr;
                }
            }
    }; // class Array

    enum AccessMethod : bool {
        WRITE,
        READ,
    };

    template <typename T>
    constexpr inline bool checkAccess(const Array<T>& a, const AccessMethod am) {
        const AccessType at = a.getAccessType();
        if (at == READ_WRITE) return true;
        if (am == WRITE) return (at == WRITE_ONLY);
        else return (at == READ_ONLY);
    }

    class Device {
        private:
            cl_platform_id platform;
            cl_device_id device;
            cl_context context;
            cl_command_queue queue;
            
        public:
            Device() {}
            Device(const Device&) = delete;
            Device(cl_platform_id pf, cl_device_id dev) : platform(pf), device(dev) {
                cl_int err;
                context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
                checkErr(err, "clCreateContext");

                constexpr cl_queue_properties props[] = {0}; // no properties
                queue = clCreateCommandQueueWithProperties(context, device, props, &err);
                checkErr(err, "clCreateCommandQueueWithProperties");
            }
            Device(Device&& other) {
                platform = other.platform;
                device = other.device;
                context = other.context;
                queue = other.queue;

                other.context = nullptr;
                other.queue = nullptr;
            }
            
            cl_platform_id& getPlatform() {return platform;}
            cl_device_id& getDevice() {return device;}
            cl_context& getContext() {return context;}
            cl_command_queue& getQueue() {return queue;}

            Device& operator=(const Device&) = delete;
            Device& operator=(Device&& other) {
                if (this != &other) {
                    // Release existing
                    if (queue) clReleaseCommandQueue(queue);
                    if (context) clReleaseContext(context);

                    platform = other.platform;
                    device = other.device;
                    context = other.context;
                    queue = other.queue;

                    other.context = nullptr;
                    other.queue = nullptr;
                }
                return *this;
            }

            #pragma region // operations`;
    
    let _opType;
    let _numType;

    for (let i = 0; i < 4; i++) { // for each opType
        _opType = opType[i];
        source +=  "\n                #pragma region // " + opMeta[_opType].name;

        for (let j = 0; j < 11; j++) { // for each numType
            _numType = numType[j];
            if (_numType == "FLOAT16") continue; // unsupported

            source += ""
                + "\n                    void " + opMeta[_opType].name + "(Array<" + numMeta[_numType].numName + ">& a, Array<" + numMeta[_numType].numName + ">& b, Array<" + numMeta[_numType].numName + ">& c) {"
                + "\n                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {"
                + "\n                            throw std::runtime_error(\"invalid Array access permissions\");"
                + "\n                        }"
                + "\n                        "
                + "\n                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {"
                + "\n                            throw std::runtime_error(\"all Arrays must be the same size\");"
                + "\n                        }"
                + "\n                        "
                + "\n                        cl_int err;"
                + "\n                        "
                + "\n                        cl_mem& aMem = a.getMem();"
                + "\n                        cl_mem& bMem = b.getMem();"
                + "\n                        cl_mem& cMem = c.getMem();"
                + "\n                        "
                + "\n                        const char* kernString = makeKernelFunction(\"" + opMeta[_opType].name + "_" + numMeta[_numType].className + "\", \"" + numMeta[_numType].numName + "\", \'" + opMeta[_opType].op + "\');"
                + "\n                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);"
                + "\n                        checkErr(err, \"clCreateProgramWithSource\");"
                + "\n                        "
                + "\n                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);"
                + "\n                        checkErr(err, \"clBuildProgram\");"
                + "\n                        "
                + "\n                        cl_kernel kernel = clCreateKernel(program, \""  + opMeta[_opType].name + "_" + numMeta[_numType].className + "\", &err);"
                + "\n                        checkErr(err, \"clCreateKernel\");"
                + "\n                        "
                + "\n                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);"
                + "\n                        checkErr(err, \"clSetKernelArg a\");"
                + "\n                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);"
                + "\n                        checkErr(err, \"clSetKernelArg b\");"
                + "\n                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);"
                + "\n                        checkErr(err, \"clSetKernelArg c\");"
                + "\n                        const size_t s = c.getSize();"
                + "\n                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);"
                + "\n                        checkErr(err, \"clSetKernelArg s\");"
                + "\n                        "
                + "\n                        size_t global_work_size = c.getSize();"
                + "\n                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);"
                + "\n                        checkErr(err, \"clEnqueueNDRangeKernel\");"
                + "\n                        "
                + "\n                        clReleaseKernel(kernel);"
                + "\n                        clReleaseProgram(program);"
                + "\n                    }"
                + "\n"
            ;
        }

        source += ""
            + "                #pragma endregion // " + opMeta[_opType].name
            + "\n"
        ;
    }

    source += `            #pragma endregion // operations

            ~Device() {
                if (queue) {
                    clReleaseCommandQueue(queue);
                    queue = nullptr;
                }

                if (context) {
                    clReleaseContext(context);
                    context = nullptr;
                }
            }
    }; // class Device

    // has to be defined after Device class definition
    template <typename T>
    Array<T>::Array(Device& dev, AccessType acc, const std::vector<T>& dat) : device(dev), access(acc), _size(dat.size()) {
        cl_int err;
        data = clCreateBuffer(device.getContext(), access, sizeof(T) * dat.size(), (void*)dat.data(), &err);
        checkErr(err, "clCreateBuffer");
    }
    
    template <typename T>
    template <size_t S>
    Array<T>::Array(Device& dev, AccessType acc, const std::array<T, S>& dat) : device(dev), access(acc), _size(S) {
        cl_int err;
        data = clCreateBuffer(device.getContext(), access, sizeof(T) * S, (void*)dat.data(), &err);
        checkErr(err, "clCreateBuffer");
    }
    
    template <typename T>
    Array<T>::Array(Device& dev, AccessType acc, const T* dat, const size_t s) : device(dev), access(acc), _size(s) {
        cl_int err;
        data = clCreateBuffer(device.getContext(), access, sizeof(T) * s, (void*)dat, &err);
        checkErr(err, "clCreateBuffer");
    }

    template <typename T>
    void Array<T>::read(std::vector<T>& v) {
        cl_int err;
        v = std::vector<T>(_size);
        err = clEnqueueReadBuffer(device.getQueue(), data, CL_TRUE, 0, sizeof(T) * _size, v.data(), 0, nullptr, nullptr);
        checkErr(err, "clEnqueueReadBuffer");
    }
    
    template <typename T>
    template <size_t S>
    void Array<T>::read(std::array<T, S>& a) {
        if (S != _size) throw std::runtime_error("read target array size mismatch");
        cl_int err;
        err = clEnqueueReadBuffer(device.getQueue(), data, CL_TRUE, 0, sizeof(T) * _size, a.data(), 0, nullptr, nullptr);
        checkErr(err, "clEnqueueReadBuffer");
    }

    template <typename T>
    void Array<T>::read(T* dat, const size_t s) {
        if (s != _size) throw std::runtime_error("read target array size mismatch");
        cl_int err;
        err = clEnqueueReadBuffer(device.getQueue(), data, CL_TRUE, 0, sizeof(T) * _size, dat, 0, nullptr, nullptr);
        checkErr(err, "clEnqueueReadBuffer");
    }
} // namespace ezcl`;

    fs.writeFile(sourcePath, source, (err) => {
        if (err) console.error(err);
    });
}

const sourcePath = "../ezcl.hpp";

make(sourcePath);