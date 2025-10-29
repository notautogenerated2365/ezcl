A header for simplifying OpenCL operations on vectors and arrays in C++.

+-- make/               contains program I used to make the ezcl header file
    +-- make.js         the program
    +-- arrays.cjs      contains data for the make program
    +-- objects.cjs     contains data for the make program
+-- ezcl.hpp            the header
+-- example.cpp         an example usage of ezcl

This header defines functions for addition, subtraction, multiplication, and division
of 8 to 64-bit signed/unsigned integers and 32 to 64-bit floats.

This is quite a high-level wrapper around OpenCL, you hardly have to do anything
compared to a typical OpenCL implementation.

Just like any other OpenCL implementation, you have to link to an OpenCL library

Usage:
namespace ezcl {
    class DeviceId {
        Wraps cl_device_id to more easily pick an OpenCL device.

        DeviceId() {
            Default constructor.
        }
        DeviceId(cl_device_id) {
            Initializes a DeviceId with a cl_device_id.
        }

        std::string name() const {
            Returns CL device name.
        }
        std::string vendor() const {
            Returns CL device vendor.
        }
        std::string version() const {
            Returns CL device version.
        }

        cl_device_id id() const {
            Returns the wrapped cl_device_id.
        }
        cl_device_type type() const {
            Returns CL device type.
        }
        cl_uint computeUnits() const {
            Returns reported CL device compute units.
        }
        cl_ulong memSize() const {
            Returns CL device global memory.
        }
            
        std::string typeString() const {
            Returns the CL device type as a std::string instead of cl_device_type.
        }

        operator cl_device_id() const {
            Returns the wrapped cl_device_id.
        }
    }

    class PlatformId {
        Wraps cl_platform_id to more easily pick an OpenCL platform.

        PlatformId() {
            Default constructor.
        }
        PlatformId(cl_platform_id) {
            Initializes a PlatformId with a cl_platform_id.
        }
            
        cl_platform_id id() const {
            Returns the wrapped cl_platform_id.
        }
        const std::vector<DeviceId>& getDevices() const {
            Returns a premade std::vector of all the DeviceId's belonging to this PlatformId.
        }

        std::string name() const {
            Returns CL platform name
        }
        std::string vendor() const {
            Returns CL platform vendor
        }
        std::string version() const {
            Returns CL platform version
        }
        std::string profile() const {
            Returns CL platform profile
        }

        operator cl_platform_id() const {
            Returns the wrapped cl_platform_id.
        }
    }

    inline std::vector<PlatformId> getPlatforms() {
        Returns a premade std::vector of all available CL platforms.
        Iterating through this vector, and the vector of DeviceId's
        provided by each PlatformId, can help you pick the best target device.
    }

    enum AccessType {
        An enumeration to determine what an ezcl Device can and cannot do to
        to an ezcl Array. See later for more details on Device and Array.
        Options:
            READ_WRITE,
            READ_ONLY,
            WRITE_ONLY
    }

    template <typename T>
    class Array {
        Wraps cl_mem, and represents an array of data on an ezcl Device.
        See later for ezcl Device.
        Upon construction of an Array, data is allocated on the device.
        Upon destruction, data is deallocated.

        Array() = delete;
        Array(const Array&) = delete;

        Array(Device&, AccessType, const std::vector<T>&) {
            Initializes an Array on an ezcl Device from an std::vector.
        }
        template <size_t S>
        Array(Device&, AccessType, const std::array<T, S>&) {
            Initializes an Array on an ezcl Device from an std::array.
        }
        Array(Device&, AccessType, const T*, const size_t) {
            Initializes an Array on an ezcl Device from a C-style array.
        }
        Array(Array&&) {
            Used for safely constructing an Array from another Array.
        }
            
        const Device& getDevice() const {
            Return the ezcl Device this Array is allocated on.
        }
        cl_mem& getMem() {
            Return the wrapped cl_mem.
        }
        AccessType getAccessType() const {
            Read the AccessType of the Array.
            Remember that this only effects what the ezcl Device can do to the Array,
            not what you can do to it.
        }
        size_t getSize() const {
            Return the size of the Array.
        }
        size_t size() const {
            Also return the size, for consistency.
        }

        void read(std::vector<T>&) {
            Read the contents of the Array back from the device into
            a std::vector.
        }
        template <size_t S>
        void read(std::array<T, S>&) {
            Read the contents of the Array back from the device into
            a std::array.
        }
        void read(T*, const size_t) {
            Read the contents of the Array back from the device into
            a C-style array.
        }
        
        Array& operator=(const Array&) = delete;
        Array& operator=(Array&&) {
            Used to safely assign this Array from another Array.
        }

        ~Array() {
            Cleans up on the ezcl Device when the Array is destroyed.
        }
    }

    class Device {
        Wraps an OpenCL device, on which Arrays can be allocated and
        mathemetical operations can be completed.

        Device() {
            Default constructor.
        }
        Device(const Device&) = delete;
        Device(cl_platform_id, cl_device_id) {
            Constructs a Device with a cl_platform_id
        }
        Device(Device&&) {
            Safely constructs a Device from another Device
        }
            
        const cl_platform_id& getPlatform() {
            Return the cl_platform_id of this Device.
        }
        const cl_device_id& getDevice() {
            Return the cl_device_id of this Device.
        }
        const cl_context& getContext() {
            Return the cl_context for this Device.
        }
        const cl_command_queue& getQueue() {
            Return the cl_command_queue for this Device.
        }

        Device& operator=(const Device&) = delete;
        Device& operator=(Device&&) {
            Safely assign a Device to another Device.
        }

        Here is where the operation functions are defined.
        There are four main functions, add, sub, mul, and div,
        with a multitude of overloads for Arrays of each supported
        underlying datatype.
        Their signatures are as follows:
            void OPNAME(Array<TYPE>&, Array<TYPE>&, Array<TYPE>&)
        where OPNAME is the name of the operation (add, sub, mul, or div) and
        TYPE is the name of the underlying type (int, float, double, etc.).
        The first two Arrays are the operands and the third Array is the result.
        The operands must have READ_WRITE or READ_ONLY AccessType,
        and the result must have READ_WRITE or WRITE_ONLY AccessType.
            
        ~Device() {
            Safely cleans up a Device.
        }
    }
}

There are a number of smaller helper functions defined, but are intended for internal use,
and you likely won't need them especially for simple use cases.

See example.cpp for some of these functions in action.

By default, CL kernel/program caching is used by transparently storing the kernels and programs in
unordered_maps associated with each ezcl Device. This can be disabled during compile time by
defining EZCL_NO_CACHE before including the header.