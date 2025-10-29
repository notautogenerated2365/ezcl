#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace ezcl {
    inline std::string makeKernelFunction(const char* name, const char* typeName, const char opOperator) {
        std::ostringstream function;

        function
            << "__kernel void " << name << "(__global const " << typeName << "* a, __global const " << typeName << "* b, __global " << typeName << "* c, const ulong s) {"
            << "\n    int gid = get_global_id(0);"
            << "\n    if (gid < s) c[gid] = a[gid] " << opOperator << " b[gid];"
            << "\n}"
        ;
        
        return function.str();
    }

    inline void checkErr(cl_int err, const char* name) {
        if (err != CL_SUCCESS) {
            throw std::runtime_error(std::string("Error: ") + std::string(name) + std::string(" (") + std::to_string(err) + std::string(")\n"));
        }
    }

    class DeviceId {
        private:
            cl_device_id _id;

            std::string getInfoString(cl_device_info param) const {
                size_t size = 0;
                clGetDeviceInfo(_id, param, 0, nullptr, &size);

                std::string value(size, '\0');
                clGetDeviceInfo(_id, param, size, value.data(), nullptr);

                if (!value.empty() && value.back() == '\0') {
                    value.pop_back();
                }
                
                return value;
            }
        
        public:
            DeviceId() {}
            DeviceId(cl_device_id i) : _id(i) {}

            std::string name() const {
                return getInfoString(CL_DEVICE_NAME);
            }
            std::string vendor() const {
                return getInfoString(CL_DEVICE_VENDOR);
            }
            std::string version() const {
                return getInfoString(CL_DEVICE_VERSION);
            }

            cl_device_id id() const {
                return _id;
            }
            cl_device_type type() const {
                cl_device_type t;
                clGetDeviceInfo(_id, CL_DEVICE_TYPE, sizeof(t), &t, nullptr);
                return t;
            }
            cl_uint computeUnits() const {
                cl_uint cu;
                clGetDeviceInfo(_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
                return cu;
            }
            cl_ulong memSize() const {
                cl_ulong mem;
                clGetDeviceInfo(_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
                return mem;
            }
            
            std::string typeString() const {
                cl_device_type t = type();
                if (t & CL_DEVICE_TYPE_GPU)  return "GPU";
                if (t & CL_DEVICE_TYPE_CPU)  return "CPU";
                if (t & CL_DEVICE_TYPE_ACCELERATOR) return "Accelerator";
                if (t & CL_DEVICE_TYPE_DEFAULT) return "Default";
                return "Unknown";
            }

            operator cl_device_id() const {
                return _id;
            }
    };

    class PlatformId {
        private:
            cl_platform_id _id;
            std::vector<DeviceId> devices;

            void queryDevices() {
                cl_uint numDevices = 0;

                if (
                    clGetDeviceIDs(_id, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices) != CL_SUCCESS ||
                    numDevices == 0
                ) return;
                
                std::vector<cl_device_id> deviceIds(numDevices);
                clGetDeviceIDs(_id, CL_DEVICE_TYPE_ALL, numDevices, deviceIds.data(), nullptr);
                devices.reserve(numDevices);

                for (auto d : deviceIds) {
                    devices.emplace_back(d);
                }
            }

            std::string getInfoString(cl_platform_info param) const {
                size_t size = 0;
                clGetPlatformInfo(_id, param, 0, nullptr, &size);

                std::string value(size, '\0');
                clGetPlatformInfo(_id, param, size, value.data(), nullptr);

                if (!value.empty() && value.back() == '\0') value.pop_back();
                return value;
            }

        public:
            PlatformId() {}
            PlatformId(cl_platform_id pid) : _id(pid) {
                queryDevices();
            }
            
            cl_platform_id id() const {return _id;}
            const std::vector<DeviceId>& getDevices() const {return devices;}

            std::string name() const {
                return getInfoString(CL_PLATFORM_NAME);
            }
            std::string vendor() const {
                return getInfoString(CL_PLATFORM_VENDOR);
            }
            std::string version() const {
                return getInfoString(CL_PLATFORM_VERSION);
            }
            std::string profile() const {
                return getInfoString(CL_PLATFORM_PROFILE);
            }

            operator cl_platform_id() const {
                return _id;
            }
    };

    inline std::vector<PlatformId> getPlatforms() {
        cl_uint numPlatforms = 0;
        cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);

        if (err != CL_SUCCESS || numPlatforms == 0) {
            throw std::runtime_error("No OpenCL platforms found.");
        }

        std::vector<cl_platform_id> platformIds(numPlatforms);
        clGetPlatformIDs(numPlatforms, platformIds.data(), nullptr);

        std::vector<PlatformId> platforms;
        platforms.reserve(numPlatforms);

        for (auto pid : platformIds) {
            platforms.emplace_back(pid);
        }

        return platforms;
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
            size_t size_;

        public:
            Array() = delete;
            Array(const Array&) = delete;

            // has to be defined after Device class definition
            Array(Device& dev, AccessType acc, const std::vector<T>& dat);
            template <size_t S>
            Array(Device& dev, AccessType acc, const std::array<T, S>& dat);
            Array(Device& dev, AccessType acc, const T* dat, const size_t s);
            Array(Array&& other) : device(other.device), data(other.data), access(other.access), size_(other.size_) {
                other.data = nullptr;
                other.size_ = 0;
            }
            
            const Device& getDevice() const {return device;}
            cl_mem& getMem() {return data;}
            AccessType getAccessType() const {return access;}
            size_t getSize() const {return size_;}
            size_t size() const {return size_;}

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
                    size_ = other.size_;
                    other.data = nullptr;
                    other.size_ = 0;
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

            #ifndef EZCL_NO_CACHE
                std::unordered_map<std::string, cl_program> programCache;
                std::unordered_map<std::string, cl_kernel> kernelCache;
            #endif
            
            cl_program buildProgram(const std::string& src, const std::string& key) {
                cl_int err;
                
                #ifndef EZCL_NO_CACHE
                    auto it = programCache.find(key);
                    if (it != programCache.end()) return it->second;
                #endif

                const char* csrc = src.c_str();
                cl_program program = clCreateProgramWithSource(context, 1, &csrc, nullptr, &err);
                checkErr(err, "clCreateProgramWithSource");
                err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
                checkErr(err, "clBuildProgram");

                #ifndef EZCL_NO_CACHE
                    programCache[key] = program;
                #endif

                return program;
            }

            cl_kernel getKernel(const std::string& key, cl_program program) {
                cl_int err;

                #ifndef EZCL_NO_CACHE
                    auto it = kernelCache.find(key);
                    if (it != kernelCache.end()) return it->second;
                #endif

                cl_kernel kernel = clCreateKernel(program, key.c_str(), &err);
                checkErr(err, "clCreateKernel");

                #ifndef EZCL_NO_CACHE
                    kernelCache[key] = kernel;
                #endif

                return kernel;
            }

            void launchKernel(cl_kernel kernel, cl_mem& a, cl_mem& b, cl_mem& c, size_t size) {
                cl_int err;
                err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a);
                checkErr(err, "clSetKernelArg a");
                err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b);
                checkErr(err, "clSetKernelArg b");
                err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c);
                checkErr(err, "clSetKernelArg c");
                err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &size);
                checkErr(err, "clSetKernelArg s");

                size_t global_work_size = size;
                err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
                checkErr(err, "clEnqueueNDRangeKernel");
            }
            
        public:
            Device() : platform(nullptr), device(nullptr), context(nullptr), queue(nullptr) {}
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
            
            const cl_platform_id& getPlatform() {return platform;}
            const cl_device_id& getDevice() {return device;}
            const cl_context& getContext() {return context;}
            const cl_command_queue& getQueue() {return queue;}

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

                        const std::string kernelKey = "add_int8";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "char", '+');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void add(Array<short>& a, Array<short>& b, Array<short>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "add_int16";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "short", '+');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void add(Array<int>& a, Array<int>& b, Array<int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "add_int32";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "int", '+');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void add(Array<long long int>& a, Array<long long int>& b, Array<long long int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "add_int64";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "long long int", '+');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void add(Array<unsigned char>& a, Array<unsigned char>& b, Array<unsigned char>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "add_uint8";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "unsigned char", '+');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void add(Array<unsigned short>& a, Array<unsigned short>& b, Array<unsigned short>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "add_uint16";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "unsigned short", '+');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void add(Array<unsigned int>& a, Array<unsigned int>& b, Array<unsigned int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "add_uint32";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "unsigned int", '+');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void add(Array<unsigned long long int>& a, Array<unsigned long long int>& b, Array<unsigned long long int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "add_uint64";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "unsigned long long int", '+');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void add(Array<float>& a, Array<float>& b, Array<float>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "add_float32";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "float", '+');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void add(Array<double>& a, Array<double>& b, Array<double>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "add_float64";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "double", '+');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
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

                        const std::string kernelKey = "sub_int8";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "char", '-');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void sub(Array<short>& a, Array<short>& b, Array<short>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "sub_int16";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "short", '-');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void sub(Array<int>& a, Array<int>& b, Array<int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "sub_int32";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "int", '-');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void sub(Array<long long int>& a, Array<long long int>& b, Array<long long int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "sub_int64";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "long long int", '-');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void sub(Array<unsigned char>& a, Array<unsigned char>& b, Array<unsigned char>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "sub_uint8";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "unsigned char", '-');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void sub(Array<unsigned short>& a, Array<unsigned short>& b, Array<unsigned short>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "sub_uint16";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "unsigned short", '-');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void sub(Array<unsigned int>& a, Array<unsigned int>& b, Array<unsigned int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "sub_uint32";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "unsigned int", '-');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void sub(Array<unsigned long long int>& a, Array<unsigned long long int>& b, Array<unsigned long long int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "sub_uint64";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "unsigned long long int", '-');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void sub(Array<float>& a, Array<float>& b, Array<float>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "sub_float32";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "float", '-');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void sub(Array<double>& a, Array<double>& b, Array<double>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "sub_float64";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "double", '-');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
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

                        const std::string kernelKey = "mul_int8";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "char", '*');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void mul(Array<short>& a, Array<short>& b, Array<short>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "mul_int16";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "short", '*');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void mul(Array<int>& a, Array<int>& b, Array<int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "mul_int32";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "int", '*');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void mul(Array<long long int>& a, Array<long long int>& b, Array<long long int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "mul_int64";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "long long int", '*');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void mul(Array<unsigned char>& a, Array<unsigned char>& b, Array<unsigned char>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "mul_uint8";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "unsigned char", '*');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void mul(Array<unsigned short>& a, Array<unsigned short>& b, Array<unsigned short>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "mul_uint16";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "unsigned short", '*');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void mul(Array<unsigned int>& a, Array<unsigned int>& b, Array<unsigned int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "mul_uint32";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "unsigned int", '*');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void mul(Array<unsigned long long int>& a, Array<unsigned long long int>& b, Array<unsigned long long int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "mul_uint64";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "unsigned long long int", '*');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void mul(Array<float>& a, Array<float>& b, Array<float>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "mul_float32";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "float", '*');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void mul(Array<double>& a, Array<double>& b, Array<double>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "mul_float64";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "double", '*');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
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

                        const std::string kernelKey = "div_int8";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "char", '/');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void div(Array<short>& a, Array<short>& b, Array<short>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "div_int16";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "short", '/');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void div(Array<int>& a, Array<int>& b, Array<int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "div_int32";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "int", '/');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void div(Array<long long int>& a, Array<long long int>& b, Array<long long int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "div_int64";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "long long int", '/');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void div(Array<unsigned char>& a, Array<unsigned char>& b, Array<unsigned char>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "div_uint8";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "unsigned char", '/');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void div(Array<unsigned short>& a, Array<unsigned short>& b, Array<unsigned short>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "div_uint16";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "unsigned short", '/');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void div(Array<unsigned int>& a, Array<unsigned int>& b, Array<unsigned int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "div_uint32";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "unsigned int", '/');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void div(Array<unsigned long long int>& a, Array<unsigned long long int>& b, Array<unsigned long long int>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "div_uint64";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "unsigned long long int", '/');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void div(Array<float>& a, Array<float>& b, Array<float>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "div_float32";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "float", '/');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
                    }
                
                    void div(Array<double>& a, Array<double>& b, Array<double>& c) {
                        if (!checkAccess(a, READ) || !checkAccess(b, READ) || !checkAccess(c, WRITE)) {
                            throw std::runtime_error("invalid Array access permissions");
                        }

                        if ((a.getSize() != c.getSize()) || (b.getSize() != c.getSize())) {
                            throw std::runtime_error("all Arrays must be the same size");
                        }

                        const std::string kernelKey = "div_float64";
                        const std::string kernString = makeKernelFunction(kernelKey.c_str(), "double", '/');
                        
                        cl_program program = buildProgram(kernString, kernelKey);
                        cl_kernel kernel = getKernel(kernelKey, program);
                        launchKernel(kernel, a.getMem(), b.getMem(), c.getMem(), c.getSize());

                        #ifdef EZCL_NO_CACHE
                            clReleaseKernel(kernel);
                            clReleaseProgram(program);
                        #endif
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

                #ifndef EZCL_NO_CACHE
                    for (auto& kv : kernelCache)
                        clReleaseKernel(kv.second);
                    kernelCache.clear();

                    for (auto& kv : programCache)
                        clReleaseProgram(kv.second);
                    programCache.clear();
                #endif
            }
    }; // class Device

    // has to be defined after Device class definition
    template <typename T>
    Array<T>::Array(Device& dev, AccessType acc, const std::vector<T>& dat) : device(dev), access(acc), size_(dat.size()) {
        cl_int err;
        data = clCreateBuffer(device.getContext(), access, sizeof(T) * dat.size(), (void*)dat.data(), &err);
        checkErr(err, "clCreateBuffer");
    }
    
    template <typename T>
    template <size_t S>
    Array<T>::Array(Device& dev, AccessType acc, const std::array<T, S>& dat) : device(dev), access(acc), size_(S) {
        cl_int err;
        data = clCreateBuffer(device.getContext(), access, sizeof(T) * S, (void*)dat.data(), &err);
        checkErr(err, "clCreateBuffer");
    }
    
    template <typename T>
    Array<T>::Array(Device& dev, AccessType acc, const T* dat, const size_t s) : device(dev), access(acc), size_(s) {
        cl_int err;
        data = clCreateBuffer(device.getContext(), access, sizeof(T) * s, (void*)dat, &err);
        checkErr(err, "clCreateBuffer");
    }

    template <typename T>
    void Array<T>::read(std::vector<T>& v) {
        cl_int err;
        v = std::vector<T>(size_);
        err = clEnqueueReadBuffer(device.getQueue(), data, CL_TRUE, 0, sizeof(T) * size_, v.data(), 0, nullptr, nullptr);
        checkErr(err, "clEnqueueReadBuffer");
    }
    
    template <typename T>
    template <size_t S>
    void Array<T>::read(std::array<T, S>& a) {
        if (S != size_) throw std::runtime_error("read target array size mismatch");
        cl_int err;
        err = clEnqueueReadBuffer(device.getQueue(), data, CL_TRUE, 0, sizeof(T) * size_, a.data(), 0, nullptr, nullptr);
        checkErr(err, "clEnqueueReadBuffer");
    }

    template <typename T>
    void Array<T>::read(T* dat, const size_t s) {
        if (s != size_) throw std::runtime_error("read target array size mismatch");
        cl_int err;
        err = clEnqueueReadBuffer(device.getQueue(), data, CL_TRUE, 0, sizeof(T) * size_, dat, 0, nullptr, nullptr);
        checkErr(err, "clEnqueueReadBuffer");
    }
} // namespace ezcl