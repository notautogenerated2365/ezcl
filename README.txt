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