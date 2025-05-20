# "c:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
nvcc -std=c++17 -arch=native -O3 --expt-relaxed-constexpr --default-stream legacy main.cu -o main
