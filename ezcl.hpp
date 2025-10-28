#pragma once

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
            << "\n    int gid = get_global_id(0);"
            << "\n    if (gid < s) c[gid] = a[gid] " << opOperator << " b[gid];"
            << "\n}"
        ;
        
        return function.str().c_str();
    }

    inline void checkErr(cl_int err, const char* name) {
        if (err != CL_SUCCESS) {
            throw std::runtime_error(std::string("Error: ") + std::string(name) + std::string(" (") + std::to_string(err) + std::string(")\n"));
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

            #pragma region // operations
                #pragma region // add
                    void add(Array<char>& a, Array<char>& b, Array<char>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("add_int8", "char", '+');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "add_int8", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void add(Array<short>& a, Array<short>& b, Array<short>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("add_int16", "short", '+');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "add_int16", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void add(Array<int>& a, Array<int>& b, Array<int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("add_int32", "int", '+');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "add_int32", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void add(Array<long long int>& a, Array<long long int>& b, Array<long long int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("add_int64", "long long int", '+');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "add_int64", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void add(Array<unsigned char>& a, Array<unsigned char>& b, Array<unsigned char>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("add_uint8", "unsigned char", '+');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "add_uint8", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void add(Array<unsigned short>& a, Array<unsigned short>& b, Array<unsigned short>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("add_uint16", "unsigned short", '+');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "add_uint16", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void add(Array<unsigned int>& a, Array<unsigned int>& b, Array<unsigned int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("add_uint32", "unsigned int", '+');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "add_uint32", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void add(Array<unsigned long long int>& a, Array<unsigned long long int>& b, Array<unsigned long long int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("add_uint64", "unsigned long long int", '+');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "add_uint64", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void add(Array<float>& a, Array<float>& b, Array<float>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("add_float32", "float", '+');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "add_float32", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void add(Array<double>& a, Array<double>& b, Array<double>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("add_float64", "double", '+');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "add_float64", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }
                #pragma endregion // add

                #pragma region // sub
                    void sub(Array<char>& a, Array<char>& b, Array<char>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("sub_int8", "char", '-');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "sub_int8", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void sub(Array<short>& a, Array<short>& b, Array<short>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("sub_int16", "short", '-');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "sub_int16", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void sub(Array<int>& a, Array<int>& b, Array<int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("sub_int32", "int", '-');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "sub_int32", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void sub(Array<long long int>& a, Array<long long int>& b, Array<long long int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("sub_int64", "long long int", '-');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "sub_int64", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void sub(Array<unsigned char>& a, Array<unsigned char>& b, Array<unsigned char>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("sub_uint8", "unsigned char", '-');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "sub_uint8", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void sub(Array<unsigned short>& a, Array<unsigned short>& b, Array<unsigned short>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("sub_uint16", "unsigned short", '-');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "sub_uint16", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void sub(Array<unsigned int>& a, Array<unsigned int>& b, Array<unsigned int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("sub_uint32", "unsigned int", '-');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "sub_uint32", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void sub(Array<unsigned long long int>& a, Array<unsigned long long int>& b, Array<unsigned long long int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("sub_uint64", "unsigned long long int", '-');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "sub_uint64", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void sub(Array<float>& a, Array<float>& b, Array<float>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("sub_float32", "float", '-');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "sub_float32", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void sub(Array<double>& a, Array<double>& b, Array<double>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("sub_float64", "double", '-');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "sub_float64", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }
                #pragma endregion // sub

                #pragma region // mul
                    void mul(Array<char>& a, Array<char>& b, Array<char>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("mul_int8", "char", '*');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "mul_int8", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void mul(Array<short>& a, Array<short>& b, Array<short>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("mul_int16", "short", '*');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "mul_int16", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void mul(Array<int>& a, Array<int>& b, Array<int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("mul_int32", "int", '*');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "mul_int32", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void mul(Array<long long int>& a, Array<long long int>& b, Array<long long int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("mul_int64", "long long int", '*');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "mul_int64", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void mul(Array<unsigned char>& a, Array<unsigned char>& b, Array<unsigned char>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("mul_uint8", "unsigned char", '*');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "mul_uint8", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void mul(Array<unsigned short>& a, Array<unsigned short>& b, Array<unsigned short>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("mul_uint16", "unsigned short", '*');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "mul_uint16", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void mul(Array<unsigned int>& a, Array<unsigned int>& b, Array<unsigned int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("mul_uint32", "unsigned int", '*');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "mul_uint32", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void mul(Array<unsigned long long int>& a, Array<unsigned long long int>& b, Array<unsigned long long int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("mul_uint64", "unsigned long long int", '*');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "mul_uint64", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void mul(Array<float>& a, Array<float>& b, Array<float>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("mul_float32", "float", '*');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "mul_float32", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void mul(Array<double>& a, Array<double>& b, Array<double>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("mul_float64", "double", '*');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "mul_float64", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }
                #pragma endregion // mul

                #pragma region // div
                    void div(Array<char>& a, Array<char>& b, Array<char>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("div_int8", "char", '/');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "div_int8", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void div(Array<short>& a, Array<short>& b, Array<short>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("div_int16", "short", '/');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "div_int16", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void div(Array<int>& a, Array<int>& b, Array<int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("div_int32", "int", '/');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "div_int32", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void div(Array<long long int>& a, Array<long long int>& b, Array<long long int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("div_int64", "long long int", '/');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "div_int64", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void div(Array<unsigned char>& a, Array<unsigned char>& b, Array<unsigned char>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("div_uint8", "unsigned char", '/');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "div_uint8", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void div(Array<unsigned short>& a, Array<unsigned short>& b, Array<unsigned short>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("div_uint16", "unsigned short", '/');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "div_uint16", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void div(Array<unsigned int>& a, Array<unsigned int>& b, Array<unsigned int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("div_uint32", "unsigned int", '/');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "div_uint32", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void div(Array<unsigned long long int>& a, Array<unsigned long long int>& b, Array<unsigned long long int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("div_uint64", "unsigned long long int", '/');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "div_uint64", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void div(Array<float>& a, Array<float>& b, Array<float>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("div_float32", "float", '/');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "div_float32", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }

                    void div(Array<double>& a, Array<double>& b, Array<double>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }
                        
                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }
                        
                        cl_int err;
                        
                        cl_mem& aMem = a.getMem();
                        cl_mem& bMem = b.getMem();
                        cl_mem& cMem = c.getMem();
                        
                        const char* kernString = makeKernelFunction("div_float64", "double", '/');
                        cl_program program = clCreateProgramWithSource(context, 1, &kernString, nullptr, &err);
                        checkErr(err, "clCreateProgramWithSource");
                        
                        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                        checkErr(err, "clBuildProgram");
                        
                        cl_kernel kernel = clCreateKernel(program, "div_float64", &err);
                        checkErr(err, "clCreateKernel");
                        
                        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aMem);
                        checkErr(err, "clSetKernelArg a");
                        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bMem);
                        checkErr(err, "clSetKernelArg b");
                        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cMem);
                        checkErr(err, "clSetKernelArg c");
                        const size_t s = c.getSize();
                        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &s);
                        checkErr(err, "clSetKernelArg s");
                        
                        size_t global_work_size = c.getSize();
                        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                        checkErr(err, "clEnqueueNDRangeKernel");
                        
                        clReleaseKernel(kernel);
                        clReleaseProgram(program);
                    }
                #pragma endregion // div
            #pragma endregion // operations

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
} // namespace ezcl