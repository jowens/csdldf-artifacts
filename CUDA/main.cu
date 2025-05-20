#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cub/cub.cuh>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <stdexcept> // For std::stoul exceptions (std::invalid_argument, std::out_of_range)
#include <limits>    // For std::numeric_limits
#include <iomanip>   // For std::fixed and std::setprecision

#include "CSDLDF.cuh"

#define WARPS 8
#define UINT4_PER_THREAD 4
#define checkCudaError()                                         \
    {                                                            \
        cudaError_t err = cudaGetLastError();                    \
        if (err != cudaSuccess) {                                \
            printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        }                                                        \
    }

__global__ void InitOne(uint32_t* scan, uint32_t size) {
    const uint32_t increment = blockDim.x * gridDim.x;
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += increment) {
        scan[i] = 1;
    }
}

__global__ void ValidateInclusive(uint32_t* scan, uint32_t* errCount, uint32_t size) {
    const uint32_t increment = blockDim.x * gridDim.x;
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += increment) {
        if (scan[i] != i + 1) {
            atomicAdd(&errCount[0], 1);
        }
    }
}

bool DispatchValidateInclusive(uint32_t* scanOut, uint32_t* err, uint32_t size) {
    uint32_t errCount[1] = {0xffffffff};
    cudaMemset(err, 0, sizeof(uint32_t));
    ValidateInclusive<<<256, 256>>>(scanOut, err, size);
    cudaMemcpy(&errCount, err, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return !errCount[0];
}

void WriteVectorToCSV(const std::vector<float>& data, const std::string& filename) {
    std::ofstream csvFile(filename);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    csvFile << "time\n";  // CSV header
    for (float val : data) {
        csvFile << val << "\n";
    }
    csvFile.close();
    std::cout << "CSV data written to " << filename << std::endl;
}

class CubTestRunner {
public:
    // Run the test for a given size and batch count.
    // Returns a vector of individual iteration times (in ms), excluding the warmup.
    std::vector<float> runTest(uint32_t size, uint32_t batchCount,
                               uint32_t* scanIn, uint32_t* scanOut, uint32_t* err) {
        std::vector<float> times(batchCount);
        printf("Running CUB test for size %u, batch count %u\n", size, batchCount);

        // Warm-up: determine temporary storage size.
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, scanIn, scanOut, size);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float totalTime = 0.0f;
        uint32_t testsPassed = 0;
        for (uint32_t i = 0; i <= batchCount; ++i) {
            InitOne<<<256, 256>>>(scanIn, size);
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, scanIn, scanOut, size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float millis = 0.0f;
            cudaEventElapsedTime(&millis, start, stop);

            // Discard the first (warmup) iteration.
            if (i > 0) {
                testsPassed += DispatchValidateInclusive(scanOut, err, size) ? 1 : 0;
                totalTime += millis;
                times[i - 1] = millis;  // Record individual time in milliseconds.
            }
        }

        float avgTime = totalTime / batchCount;
        float throughput = size / (avgTime / 1000.0f);  // keys/sec calculation.
        printf("CUB: Size %u: Average time: %f ms, Throughput: %E keys/sec, Tests passed: %u/%u\n",
               size, avgTime, throughput, testsPassed, batchCount);

        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return times;
    }
};

class CsdldfTestRunner {
public:
    std::vector<float> runTest(uint32_t size, uint32_t batchCount,
                               uint32_t* scanIn, uint32_t* scanOut, uint32_t* err) {
        std::vector<float> times(batchCount);
        printf("Running CSDLDF test for size %u, batch count %u\n", size, batchCount);

        const uint32_t k_csdldfThreads = WARPS * LANE_COUNT;
        const uint32_t k_partitionSize = k_csdldfThreads * UINT4_PER_THREAD * 4;
        const uint32_t k_vecSize = size / 4;  // Assumes size is a multiple of 4.

        uint32_t* index;
        uint32_t* threadBlockReduction;
        const uint32_t threadBlocks = (size + k_partitionSize - 1) / k_partitionSize;
        cudaMalloc(&index, sizeof(uint32_t));
        cudaMalloc(&threadBlockReduction, threadBlocks * 2 * sizeof(uint32_t));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float totalTime = 0.0f;
        uint32_t testsPassed = 0;
        for (uint32_t i = 0; i <= batchCount; ++i) {
            InitOne<<<256, 256>>>(scanIn, size);
            cudaMemset(index, 0, sizeof(uint32_t));
            cudaMemset(threadBlockReduction, 0, threadBlocks * 2 * sizeof(uint32_t));
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            ChainedScanDecoupledFallback::CSDLDFInclusive<WARPS, UINT4_PER_THREAD>
                <<<threadBlocks, k_csdldfThreads>>>(scanIn, scanOut, threadBlockReduction, index, k_vecSize);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float millis = 0.0f;
            cudaEventElapsedTime(&millis, start, stop);

            if (i > 0) {
                testsPassed += DispatchValidateInclusive(scanOut, err, size) ? 1 : 0;
                totalTime += millis;
                times[i - 1] = millis;
            }
        }

        float avgTime = totalTime / batchCount;
        float throughput = size / (avgTime / 1000.0f);
        printf("CSDLDF: Size %u: Average time: %f ms, Throughput: %E keys/sec, Tests passed: %u/%u\n",
               size, avgTime, throughput, testsPassed, batchCount);

        cudaDeviceSynchronize();
        cudaFree(index);
        cudaFree(threadBlockReduction);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return times;
    }
};

class MemcpyTestRunner {
public:
    // Note: The memcpy test does not require an error buffer.
    std::vector<float> runTest(uint32_t size, uint32_t batchCount,
                               uint32_t* scanIn, uint32_t* scanOut) {
        std::vector<float> times(batchCount);
        printf("Running Memcpy test for size %u, batch count %u\n", size, batchCount);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (uint32_t i = 0; i <= batchCount; ++i) {
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            cudaMemcpy(scanIn, scanOut, size * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float millis = 0.0f;
            cudaEventElapsedTime(&millis, start, stop);

            if (i > 0) {  // Skip warmup iteration.
                times[i - 1] = millis;
            }
        }

        float totalTime = 0.0f;
        for (float t : times)
            totalTime += t;
        float avgTime = totalTime / batchCount;
        float throughput = size / (avgTime / 1000.0f);
        printf("Memcpy: Size %u: Average time: %f ms, Throughput: %E keys/sec\n",
               size, avgTime, throughput);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return times;
    }
};

template <typename TestRunner>
std::vector<std::pair<uint32_t, std::vector<float>>> RunCudaTests(
    TestRunner& runner,
    uint32_t batchCount,
    uint32_t minPow,
    uint32_t maxPow,
    uint32_t* scanIn,
    uint32_t* scanOut,
    uint32_t* err) {
    std::vector<std::pair<uint32_t, std::vector<float>>> results;
    for (uint32_t i = minPow; i <= maxPow; ++i) {
        uint32_t n = 1u << i;
        std::vector<float> times = runner.runTest(n, batchCount, scanIn, scanOut, err);
        results.push_back(std::make_pair(n, times));
    }
    return results;
}

std::vector<std::pair<uint32_t, std::vector<float>>> RunCudaTestsMemcpy(
    MemcpyTestRunner& runner,
    uint32_t batchCount,
    uint32_t minPow,
    uint32_t maxPow,
    uint32_t* scanIn,
    uint32_t* scanOut) {
    std::vector<std::pair<uint32_t, std::vector<float>>> results;
    for (uint32_t i = minPow; i <= maxPow; ++i) {
        uint32_t n = 1u << i;
        std::vector<float> times = runner.runTest(n, batchCount, scanIn, scanOut);
        results.push_back(std::make_pair(n, times));
    }
    return results;
}

// Definition of the function to print usage instructions
void printUsage(const char* progName) {
    std::cerr << "\nUsage: " << progName << " [options] <testType> [record <csvName>]" << std::endl;
    std::cerr << "\nOptions (must appear before <testType>):" << std::endl;
    std::cerr << "  --testBatchSize <N> : Set test batch size (uint32_t, default: 2000)." << std::endl;
    std::cerr << "  --minPow <N>        : Set minimum power (uint32_t, default: 10)." << std::endl;
    std::cerr << "  --maxPow <N>        : Set maximum power (uint32_t, default: 25)." << std::endl;
    std::cerr << "\nRequired Argument (must follow any options):" << std::endl;
    std::cerr << "  <testType>          : Type of test. Must be one of 'cub', 'csdldf', 'memcpy'." << std::endl;
    std::cerr << "\nOptional Recording (must appear immediately after <testType>):" << std::endl;
    std::cerr << "  record <csvName>    : If 'record' is specified, the next argument is the CSV filename." << std::endl;
    std::cerr << "\nExamples:" << std::endl;
    std::cerr << "  " << progName << " cub" << std::endl;
    std::cerr << "  " << progName << " --testBatchSize 1000 --minPow 5 csdldf record results.csv" << std::endl;
    std::cerr << "  " << progName << " --maxPow 30 memcpy" << std::endl;
    std::cerr << "  " << progName << " cub record data.csv" << std::endl;
}

// Helper function to parse a string to uint32_t with range checking
bool parseUint32(const char* str_val, uint32_t& out_val, const std::string& arg_name, const char* progName) {
    try {
        unsigned long temp_val = std::stoul(str_val); // std::stoul is in <string>
        if (temp_val > std::numeric_limits<uint32_t>::max()) {
            std::cerr << "Error: Value for " << arg_name << " ('" << str_val << "') is out of range for uint32_t." << std::endl;
            printUsage(progName);
            return false;
        }
        out_val = static_cast<uint32_t>(temp_val);
        return true;
    } catch (const std::invalid_argument& ia) {
        std::cerr << "Error: Invalid number format for " << arg_name << ": '" << str_val << "'" << std::endl;
    } catch (const std::out_of_range& oor) { // This checks range for unsigned long
        std::cerr << "Error: Number out of range for " << arg_name << ": '" << str_val << "'" << std::endl;
    }
    // If any exception occurred or range check failed and printed error:
    printUsage(progName);
    return false;
}

void printGpuProperties(int deviceNum) {
    cudaDeviceProp deviceProp;
    cudaError_t error_id = cudaGetDeviceProperties(&deviceProp, deviceNum);

    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties returned " << static_cast<int>(error_id)
                  << " for device " << deviceNum << " -> " << cudaGetErrorString(error_id) << std::endl;
        return;
    }

    std::cout << "--- GPU Device " << deviceNum << " Properties ---" << std::endl;
    std::cout << "Name:                     " << deviceProp.name << std::endl;
    std::cout << "Compute Capability:       " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Total Global Memory:      " << std::fixed << std::setprecision(2)
              << static_cast<double>(deviceProp.totalGlobalMem) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Shared Memory Per Block:  " << deviceProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "Registers Per Block:      " << deviceProp.regsPerBlock << std::endl;
    std::cout << "Warp Size:                " << deviceProp.warpSize << std::endl;
    std::cout << "Max Threads Per Block:    " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Dim (x,y,z):  (" << deviceProp.maxThreadsDim[0] << ", "
              << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max Grid Size (x,y,z):    (" << deviceProp.maxGridSize[0] << ", "
              << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
    std::cout << "Clock Rate:               " << deviceProp.clockRate / 1000.0 << " MHz" << std::endl; // kHz to MHz
    std::cout << "Memory Clock Rate:        " << deviceProp.memoryClockRate / 1000.0 << " MHz" << std::endl; // kHz to MHz
    std::cout << "Memory Bus Width:         " << deviceProp.memoryBusWidth << " bits" << std::endl;
    std::cout << "L2 Cache Size:            " << deviceProp.l2CacheSize / 1024.0 << " KB" << std::endl;
    std::cout << "Multiprocessor Count:     " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "------------------------------------" << std::endl;
}

int main(int argc, char* argv[]) {
    // Allowed usage: <testType> [record [csvName]]
    // testType must be one of "cub", "csdldf", or "memcpy".

    // defaults
    uint32_t testBatchSize = 2000;
    uint32_t minPow = 10;
    uint32_t maxPow = 25;
    std::string testType;
    std::string csvName = "";
    bool record = false;

        int currentIndex = 1; // To keep track of the current argument being processed

    // --- Phase 1: Parse optional numeric arguments (--testBatchSize, --minPow, --maxPow) ---
    // These must appear before testType
    while (currentIndex < argc) {
        std::string currentArg = argv[currentIndex];
        if (currentArg == "--testBatchSize") {
            if (currentIndex + 1 < argc) {
                if (!parseUint32(argv[currentIndex + 1], testBatchSize, "--testBatchSize", argv[0])) return 1;
                currentIndex += 2; // Consumed option and its value
            } else {
                std::cerr << "Error: --testBatchSize option requires a value." << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        } else if (currentArg == "--minPow") {
            if (currentIndex + 1 < argc) {
                if (!parseUint32(argv[currentIndex + 1], minPow, "--minPow", argv[0])) return 1;
                currentIndex += 2;
            } else {
                std::cerr << "Error: --minPow option requires a value." << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        } else if (currentArg == "--maxPow") {
            if (currentIndex + 1 < argc) {
                if (!parseUint32(argv[currentIndex + 1], maxPow, "--maxPow", argv[0])) return 1;
                currentIndex += 2;
            } else {
                std::cerr << "Error: --maxPow option requires a value." << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        } else {
            // Argument is not one of the known optional numeric flags.
            // Assume it's the start of testType or an unknown argument.
            break;
        }
    }

    // --- Phase 2: Parse testType (required) ---
    if (currentIndex >= argc) {
        std::cerr << "Error: Required argument <testType> is missing." << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    testType = argv[currentIndex];
    if (testType != "cub" && testType != "csdldf" && testType != "memcpy") {
        std::cerr << "Error: Invalid <testType> specified: '" << testType << "'." << std::endl;
        std::cerr << "       <testType> must be one of: 'cub', 'csdldf', 'memcpy'." << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    currentIndex++; // Consumed testType

    // --- Phase 3: Parse 'record' and 'csvName' (optional, must follow testType) ---
    if (currentIndex < argc) { // Check if there are more arguments after testType
        std::string potentialRecordArg = argv[currentIndex];
        if (potentialRecordArg == "record") {
            currentIndex++; // Consumed "record" keyword
            record = true;
            if (currentIndex < argc) { // Check if there's a filename for csvName
                csvName = argv[currentIndex];
                currentIndex++; // Consumed csvName
            } else {
                std::cerr << "Error: 'record' option specified but CSV filename is missing." << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        }
        // If potentialRecordArg was not "record", it will be caught in Phase 4 if not consumed
    }

    // --- Phase 4: Check for any remaining unexpected arguments ---
    if (currentIndex < argc) {
        std::cerr << "Error: Unexpected argument(s) found, starting with: '" << argv[currentIndex] << "'" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // --- Output parsed values (for demonstration) ---
    std::cout << "--- Parsed Arguments ---" << std::endl;
    std::cout << "testBatchSize: " << testBatchSize << std::endl;
    std::cout << "minPow: " << minPow << std::endl;
    std::cout << "maxPow: " << maxPow << std::endl;
    std::cout << "testType: " << testType << std::endl;
    std::cout << "csvName: " << (csvName.empty() ? "[not set]" : csvName) << std::endl;
    std::cout << "------------------------" << std::endl;

    int currentDevice;
    cudaError_t currentDevError = cudaGetDevice(&currentDevice);
    if (currentDevError == cudaSuccess) {
        std::cout << "\n--- Properties for currently selected GPU (Device " << currentDevice << ") ---" << std::endl;
        printGpuProperties(currentDevice);
    } else {
        std::cerr << "cudaGetDevice returned " << static_cast<int>(currentDevError)
                  << " -> " << cudaGetErrorString(currentDevError) << std::endl;
    }

    uint32_t* err = nullptr;
    uint32_t* scan_in = nullptr;
    uint32_t* scan_out = nullptr;
    cudaMalloc(&err, sizeof(uint32_t));
    cudaMalloc(&scan_in, (1u << maxPow) * sizeof(uint32_t));
    cudaMalloc(&scan_out, (1u << maxPow) * sizeof(uint32_t));

    cudaError_t errStatus = cudaGetLastError();
    if (errStatus != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(errStatus) << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::pair<uint32_t, std::vector<float>>> results;
    if (testType == "cub") {
        CubTestRunner runner;
        results = RunCudaTests(runner, testBatchSize, minPow, maxPow, scan_in, scan_out, err);
    } else if (testType == "csdldf") {
        CsdldfTestRunner runner;
        results = RunCudaTests(runner, testBatchSize, minPow, maxPow, scan_in, scan_out, err);
    } else if (testType == "csdldf_prof") {
        testBatchSize = 50;
        minPow = 24;
        maxPow = 24;
        CsdldfTestRunner runner;
        results = RunCudaTests(runner, testBatchSize, minPow, maxPow, scan_in, scan_out, err);
    } else if (testType == "memcpy") {
        MemcpyTestRunner runner;
        results = RunCudaTestsMemcpy(runner, testBatchSize, minPow, maxPow, scan_in, scan_out);
    } else {
        std::cerr << "Unknown test type: " << testType << std::endl;
        return EXIT_FAILURE;
    }

    if (record) {
        for (const auto& result : results) {
            uint32_t size = result.first;
            const std::vector<float>& times = result.second;
            std::string filename = csvName + "_" + testType + "_Size_" + std::to_string(size) + ".csv";
            WriteVectorToCSV(times, filename);
        }
    }

    cudaFree(err);
    cudaFree(scan_in);
    cudaFree(scan_out);
    std::this_thread::sleep_for(std::chrono::seconds(0));
    return EXIT_SUCCESS;
}
