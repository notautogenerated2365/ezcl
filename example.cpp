#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <vector>
#include <iostream>
#include <cassert>
#include <string>

void checkErr(cl_int err, const char* name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << err << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

void vector_add(
    const std::vector<float>& a,
    const std::vector<float>& b,
    std::vector<float>& result,
    cl_context context,
    cl_device_id device,
    cl_command_queue queue
) {
    assert(a.size() == b.size());
    size_t vector_size = a.size();
    result.resize(vector_size);

    const char* kernel_source = R"CLC(
    __kernel void vector_add(__global const float* a, __global const float* b, __global float* result) {
        int gid = get_global_id(0);
        result[gid] = a[gid] + b[gid];
    }
    )CLC";

    cl_int err;

    // Create buffers
    cl_mem buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * vector_size, (void*)a.data(), &err);
    checkErr(err, "clCreateBuffer buffer_a");

    cl_mem buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * vector_size, (void*)b.data(), &err);
    checkErr(err, "clCreateBuffer buffer_b");

    cl_mem buffer_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * vector_size, nullptr, &err);
    checkErr(err, "clCreateBuffer buffer_result");

    // Create program from source
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, &err);
    checkErr(err, "clCreateProgramWithSource");

    // Build program
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Get build log size
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        std::string build_log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);

        std::cerr << "Build failed:\n" << build_log << "\n";
        clReleaseProgram(program);
        clReleaseMemObject(buffer_a);
        clReleaseMemObject(buffer_b);
        clReleaseMemObject(buffer_result);
        std::exit(EXIT_FAILURE);
    }

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    checkErr(err, "clCreateKernel");

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_a);
    checkErr(err, "clSetKernelArg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_b);
    checkErr(err, "clSetKernelArg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_result);
    checkErr(err, "clSetKernelArg 2");

    // Enqueue kernel execution
    size_t global_work_size = vector_size;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
    checkErr(err, "clEnqueueNDRangeKernel");

    // Read back results
    err = clEnqueueReadBuffer(queue, buffer_result, CL_TRUE, 0, sizeof(float) * vector_size, result.data(), 0, nullptr, nullptr);
    checkErr(err, "clEnqueueReadBuffer");

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(buffer_a);
    clReleaseMemObject(buffer_b);
    clReleaseMemObject(buffer_result);
}

int main() {
    cl_int err;

    // Get platform
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    checkErr(err, "clGetPlatformIDs count");
    if (num_platforms == 0) {
        std::cerr << "No OpenCL platforms found.\n";
        return 1;
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    checkErr(err, "clGetPlatformIDs");

    // Choose first platform
    cl_platform_id platform = platforms[0];

    // Get devices
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, 0, nullptr, &num_devices);
    checkErr(err, "clGetDeviceIDs count");
    if (num_devices == 0) {
        std::cerr << "No OpenCL devices found.\n";
        return 1;
    }
    std::vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, num_devices, devices.data(), nullptr);
    checkErr(err, "clGetDeviceIDs");

    cl_device_id device = devices[0];

    // Create context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkErr(err, "clCreateContext");

    // Create command queue (for OpenCL 1.2, no properties; for OpenCL 2.0+, you may want clCreateCommandQueueWithProperties)
    const cl_queue_properties props[] = { 0 };  // Null-terminated list of properties; empty here means default
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    checkErr(err, "clCreateCommandQueue");

    // Prepare data
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};
    std::vector<float> result;

    // Run kernel
    vector_add(a, b, result, context, device, queue);

    // Print results
    for (float f : result) {
        std::cout << f << " ";
    }
    std::cout << "\n"; // Expected: 5.0 7.0 9.0

    // Cleanup
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
