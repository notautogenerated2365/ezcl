run: main.exe
	.\main.exe

main.exe: ezcl.hpp main.cpp
	g++ main.cpp -L"C:\Program Files (x86)\Intel\oneAPI\compiler\2025.2\lib" -lOpenCL -o main