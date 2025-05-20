nvcc -std=c++14 -arch=native -O3  --expt-relaxed-constexpr --default-stream legacy main.cu -o main
