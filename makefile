run: example.exe
	.\example.exe

example.exe: ezcl.hpp example.cpp
	g++ example.cpp -L"C:\Program Files (x86)\Intel\oneAPI\compiler\2025.2\lib" -lOpenCL -o example.exe