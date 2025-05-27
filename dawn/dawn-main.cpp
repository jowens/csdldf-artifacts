#include <dawn/webgpu_cpp.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct GPUContext {
    wgpu::Instance instance;
    wgpu::Device device;
    wgpu::Queue queue;
    wgpu::QuerySet querySet;
};

struct ComputeShader {
    wgpu::BindGroup bindGroup;
    wgpu::ComputePipeline computePipeline;
    std::string label;
};

struct Shaders {
    ComputeShader init;
    ComputeShader csdldf;
    ComputeShader validate;
};

struct GPUBuffers {
    wgpu::Buffer info;
    wgpu::Buffer scanIn;
    wgpu::Buffer scanOut;
    wgpu::Buffer scanBump;
    wgpu::Buffer reduction;
    wgpu::Buffer timestamp;
    wgpu::Buffer readbackTimestamp;
    wgpu::Buffer readback;
    wgpu::Buffer misc;
    wgpu::Buffer misc2;
};

struct TestArgs {
    GPUContext& gpu;
    GPUBuffers& buffs;
    Shaders& shaders;
    uint32_t size;
    uint32_t workTiles;
    uint32_t warmupSize;
    uint32_t batchSize;
    uint32_t maxQueryEntries;
    bool shouldRecord = false;
};

void GetGPUContext(GPUContext* context, uint32_t reqTimeStampQueries, int flag) {
    wgpu::InstanceDescriptor instanceDescriptor{};
    instanceDescriptor.capabilities.timedWaitAnyEnable = true;
    wgpu::Instance instance = wgpu::CreateInstance(&instanceDescriptor);
    if (instance == nullptr) {
        std::cerr << "Instance creation failed!\n";
    }

    wgpu::RequestAdapterOptions options = {};
    options.powerPreference = wgpu::PowerPreference::HighPerformance;
    options.backendType = wgpu::BackendType::Undefined;  // specify as needed
    // options.backendType = wgpu::BackendType::Vulkan;  // specify as needed

    wgpu::Adapter adapter;
    std::promise<void> adaptPromise;
    instance.RequestAdapter(
        &options, wgpu::CallbackMode::AllowProcessEvents,
        [&](wgpu::RequestAdapterStatus status, wgpu::Adapter adapt, wgpu::StringView) {
            if (status == wgpu::RequestAdapterStatus::Success) {
                adapter = adapt;
            } else {
                std::cerr << "Failed to get adapter" << std::endl;
            }
            adaptPromise.set_value();
        });
    std::future<void> adaptFuture = adaptPromise.get_future();
    while (adaptFuture.wait_for(std::chrono::nanoseconds(100)) == std::future_status::timeout) {
        instance.ProcessEvents();
    }

    wgpu::AdapterInfo info{};
    adapter.GetInfo(&info);
    std::cout << "VendorID: " << std::hex << info.vendorID << std::dec << std::endl;
    std::cout << "Vendor: " << std::string(info.vendor.data, info.vendor.length) << std::endl;
    std::cout << "Architecture: " << std::string(info.architecture.data, info.architecture.length)
              << std::endl;
    std::cout << "DeviceID: " << std::hex << info.deviceID << std::dec << std::endl;
    std::cout << "Name: " << std::string(info.device.data, info.device.length) << std::endl;
    std::cout << "Driver description: "
              << std::string(info.description.data, info.description.length) << std::endl;
    std::cout << "Backend " << (info.backendType == wgpu::BackendType::Vulkan ? "vk" : "not vk")
              << std::endl;  // LOL

    std::vector<wgpu::FeatureName> reqFeatures = {
        wgpu::FeatureName::Subgroups,
        wgpu::FeatureName::TimestampQuery,
    };

    auto errorCallback = [](const wgpu::Device& device, wgpu::ErrorType type,
                            wgpu::StringView message) {
        std::cerr << "Error: " << std::string(message.data, message.length) << std::endl;
    };

    // next 13 lines: from
    // https://developer.chrome.com/blog/new-in-webgpu-120#timestamp_queries_quantization
    wgpu::DawnTogglesDescriptor toggles = {};
    std::vector<const char*> enabled_toggles;
    enabled_toggles.push_back("allow_unsafe_apis");
    if (flag & 1) {
        enabled_toggles.push_back("disable_robustness");
    }
    if (flag & 2) {
        enabled_toggles.push_back("disable_workgroup_init");
    }
    toggles.enabledToggleCount = enabled_toggles.size();
    toggles.enabledToggles = enabled_toggles.data();

    std::vector<const char*> disabled_toggles;
    disabled_toggles.push_back("timestamp_quantization");
    toggles.disabledToggleCount = disabled_toggles.size();
    toggles.disabledToggles = disabled_toggles.data();

    auto printToggles = [](const char* label, const std::vector<const char*>& vec) -> void {
        std::cout << label << ": [ ";
        for (const char* str : vec) {
            if (str != nullptr) {
                std::cout << "\"" << str << "\" ";  // Added quotes for clarity
            } else {
                std::cout << "[nullptr] ";
            }
        }
        std::cout << "]" << std::endl;
    };

    printToggles("Enabled toggles", enabled_toggles);
    printToggles("Disabled toggles", disabled_toggles);

    wgpu::DeviceDescriptor devDescriptor{};
    devDescriptor.nextInChain = &toggles;
    devDescriptor.requiredFeatures = reqFeatures.data();
    devDescriptor.requiredFeatureCount = static_cast<uint32_t>(reqFeatures.size());
    devDescriptor.SetUncapturedErrorCallback(errorCallback);

    wgpu::Device device;
    std::promise<void> devPromise;
    adapter.RequestDevice(
        &devDescriptor, wgpu::CallbackMode::AllowProcessEvents,
        [&](wgpu::RequestDeviceStatus status, wgpu::Device dev, wgpu::StringView sv) {
            if (status == wgpu::RequestDeviceStatus::Success) {
                device = dev;
            } else {
                std::cerr << "Failed to get device, error code " << (unsigned int)status
                          << std::endl
                          << std::string(sv.data, sv.length) << std::endl;
            }
            devPromise.set_value();
        });
    std::future<void> devFuture = devPromise.get_future();
    while (devFuture.wait_for(std::chrono::nanoseconds(100)) == std::future_status::timeout) {
        instance.ProcessEvents();
    }
    wgpu::Queue queue = device.GetQueue();

    wgpu::QuerySetDescriptor querySetDescriptor{};
    querySetDescriptor.label = "Timestamp Query Set";
    querySetDescriptor.count = reqTimeStampQueries;
    querySetDescriptor.type = wgpu::QueryType::Timestamp;
    wgpu::QuerySet querySet = device.CreateQuerySet(&querySetDescriptor);

    (*context).instance = instance;
    (*context).device = device;
    (*context).queue = queue;
    (*context).querySet = querySet;
}

void GetGPUBuffers(const wgpu::Device& device, GPUBuffers* buffs, uint32_t workTiles,
                   uint32_t reqTimeStampQueries, uint32_t size, uint32_t miscSize,
                   uint32_t maxReadbackSize) {
    wgpu::BufferDescriptor infoDesc = {};
    infoDesc.label = "Info";
    infoDesc.size = sizeof(uint32_t) * 4;
    infoDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer info = device.CreateBuffer(&infoDesc);

    wgpu::BufferDescriptor scanInDesc = {};
    scanInDesc.label = "Scan Input";
    scanInDesc.size = sizeof(uint32_t) * size;
    scanInDesc.usage = wgpu::BufferUsage::Storage;
    wgpu::Buffer scanIn = device.CreateBuffer(&scanInDesc);

    wgpu::BufferDescriptor scanOutDesc = {};
    scanOutDesc.label = "Scan Output";
    scanOutDesc.size = sizeof(uint32_t) * size;
    scanOutDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    wgpu::Buffer scanOut = device.CreateBuffer(&scanOutDesc);

    wgpu::BufferDescriptor scanBumpDesc = {};
    scanBumpDesc.label = "Scan Atomic Bump";
    scanBumpDesc.size = sizeof(uint32_t);
    scanBumpDesc.usage = wgpu::BufferUsage::Storage;
    wgpu::Buffer scanBump = device.CreateBuffer(&scanBumpDesc);

    wgpu::BufferDescriptor redDesc = {};
    redDesc.label = "Intermediate Reduction";
    redDesc.size = sizeof(uint32_t) * workTiles * 2;
    redDesc.usage = wgpu::BufferUsage::Storage;
    wgpu::Buffer reduction = device.CreateBuffer(&redDesc);

    wgpu::BufferDescriptor timestampDesc = {};
    timestampDesc.label = "Timestamp";
    timestampDesc.size = sizeof(uint64_t) * reqTimeStampQueries;
    timestampDesc.usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc;
    wgpu::Buffer timestamp = device.CreateBuffer(&timestampDesc);

    wgpu::BufferDescriptor timestampReadDesc = {};
    timestampReadDesc.label = "Timestamp Readback";
    timestampReadDesc.size = sizeof(uint64_t) * reqTimeStampQueries;
    timestampReadDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer timestampReadback = device.CreateBuffer(&timestampReadDesc);

    wgpu::BufferDescriptor readbackDesc = {};
    readbackDesc.label = "Main Readback";
    readbackDesc.size = sizeof(uint32_t) * maxReadbackSize;
    readbackDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer readback = device.CreateBuffer(&readbackDesc);

    wgpu::BufferDescriptor miscDesc = {};
    miscDesc.label = "Miscellaneous";
    miscDesc.size = sizeof(uint32_t) * miscSize;
    miscDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    wgpu::Buffer misc = device.CreateBuffer(&miscDesc);

    wgpu::BufferDescriptor misc2Desc = {};
    misc2Desc.label = "Miscellaneous 2";
    misc2Desc.size = sizeof(uint32_t) * workTiles;
    misc2Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    wgpu::Buffer misc2 = device.CreateBuffer(&misc2Desc);

    (*buffs).info = info;
    (*buffs).scanIn = scanIn;
    (*buffs).scanOut = scanOut;
    (*buffs).scanBump = scanBump;
    (*buffs).reduction = reduction;
    (*buffs).timestamp = timestamp;
    (*buffs).readbackTimestamp = timestampReadback;
    (*buffs).readback = readback;
    (*buffs).misc = misc;
    (*buffs).misc2 = misc2;
}

// For simplicity we will use the same brind group and layout for all kernels
void GetComputeShaderPipeline(const wgpu::Device& device, const GPUBuffers& buffs,
                              ComputeShader* cs, const char* entryPoint,
                              const wgpu::ShaderModule& module, const std::string& csLabel) {
    auto makeLabel = [&](const std::string& suffix) -> std::string { return csLabel + suffix; };

    wgpu::BindGroupLayoutEntry bglInfo = {};
    bglInfo.binding = 0;
    bglInfo.visibility = wgpu::ShaderStage::Compute;
    bglInfo.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry bglScanIn = {};
    bglScanIn.binding = 1;
    bglScanIn.visibility = wgpu::ShaderStage::Compute;
    bglScanIn.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry bglScanOut = {};
    bglScanOut.binding = 2;
    bglScanOut.visibility = wgpu::ShaderStage::Compute;
    bglScanOut.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry bglScanBump = {};
    bglScanBump.binding = 3;
    bglScanBump.visibility = wgpu::ShaderStage::Compute;
    bglScanBump.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry bglReduction = {};
    bglReduction.binding = 4;
    bglReduction.visibility = wgpu::ShaderStage::Compute;
    bglReduction.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry bglMisc = {};
    bglMisc.binding = 5;
    bglMisc.visibility = wgpu::ShaderStage::Compute;
    bglMisc.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry bglMisc2 = {};
    bglMisc2.binding = 6;
    bglMisc2.visibility = wgpu::ShaderStage::Compute;
    bglMisc2.buffer.type = wgpu::BufferBindingType::Storage;

    std::vector<wgpu::BindGroupLayoutEntry> bglEntries{bglInfo,     bglScanIn,    bglScanOut,
                                                       bglScanBump, bglReduction, bglMisc};

    wgpu::BindGroupLayoutDescriptor bglDesc = {};
    bglDesc.label = makeLabel("Bind Group Layout").c_str();
    bglDesc.entries = bglEntries.data();
    bglDesc.entryCount = static_cast<uint32_t>(bglEntries.size());
    wgpu::BindGroupLayout bgl = device.CreateBindGroupLayout(&bglDesc);

    wgpu::BindGroupEntry bgInfo = {};
    bgInfo.binding = 0;
    bgInfo.buffer = buffs.info;
    bgInfo.size = buffs.info.GetSize();

    wgpu::BindGroupEntry bgScanIn = {};
    bgScanIn.binding = 1;
    bgScanIn.buffer = buffs.scanIn;
    bgScanIn.size = buffs.scanIn.GetSize();

    wgpu::BindGroupEntry bgScanOut = {};
    bgScanOut.binding = 2;
    bgScanOut.buffer = buffs.scanOut;
    bgScanOut.size = buffs.scanOut.GetSize();

    wgpu::BindGroupEntry bgScanBump = {};
    bgScanBump.binding = 3;
    bgScanBump.buffer = buffs.scanBump;
    bgScanBump.size = buffs.scanBump.GetSize();

    wgpu::BindGroupEntry bgReduction = {};
    bgReduction.binding = 4;
    bgReduction.buffer = buffs.reduction;
    bgReduction.size = buffs.reduction.GetSize();

    wgpu::BindGroupEntry bgMisc = {};
    bgMisc.binding = 5;
    bgMisc.buffer = buffs.misc;
    bgMisc.size = buffs.misc.GetSize();

    wgpu::BindGroupEntry bgMisc2 = {};
    bgMisc2.binding = 6;
    bgMisc2.buffer = buffs.misc2;
    bgMisc2.size = buffs.misc2.GetSize();

    std::vector<wgpu::BindGroupEntry> bgEntries{bgInfo,     bgScanIn,    bgScanOut,
                                                bgScanBump, bgReduction, bgMisc};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.entries = bgEntries.data();
    bindGroupDesc.entryCount = static_cast<uint32_t>(bgEntries.size());
    bindGroupDesc.layout = bgl;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);

    wgpu::PipelineLayoutDescriptor pipeLayoutDesc = {};
    pipeLayoutDesc.label = makeLabel("Pipeline Layout").c_str();
    pipeLayoutDesc.bindGroupLayoutCount = 1;
    pipeLayoutDesc.bindGroupLayouts = &bgl;
    wgpu::PipelineLayout pipeLayout = device.CreatePipelineLayout(&pipeLayoutDesc);

    wgpu::ComputeState compState = {};
    compState.entryPoint = entryPoint;
    compState.module = module;

    wgpu::ComputePipelineDescriptor compPipeDesc = {};
    compPipeDesc.label = makeLabel("Compute Pipeline").c_str();
    compPipeDesc.layout = pipeLayout;
    compPipeDesc.compute = compState;
    wgpu::ComputePipeline compPipeline = device.CreateComputePipeline(&compPipeDesc);

    (*cs).bindGroup = bindGroup;
    (*cs).computePipeline = compPipeline;
    (*cs).label = csLabel;
}

std::string ReadWGSL(const std::string& path, const std::vector<std::string>& pseudoArgs) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return "";
    }

    std::stringstream buffer;
    for (size_t i = 0; i < pseudoArgs.size(); ++i) {
        buffer << pseudoArgs[i] << "\n";
    }
    buffer << file.rdbuf();
    file.close();
    return buffer.str();
}

void CreateShaderFromSource(const GPUContext& gpu, const GPUBuffers& buffs, ComputeShader* cs,
                            const char* entryPoint, const std::string& path,
                            const std::string& csLabel,
                            const std::vector<std::string>& pseudoArgs) {
    wgpu::ShaderSourceWGSL wgslSource = {};
    std::string source = ReadWGSL(path, pseudoArgs);
    wgslSource.code = source.c_str();
    wgpu::ShaderModuleDescriptor desc = {};
    desc.nextInChain = &wgslSource;
    wgpu::ShaderModule mod = gpu.device.CreateShaderModule(&desc);
    std::promise<void> promise;
    mod.GetCompilationInfo(
        wgpu::CallbackMode::AllowProcessEvents,
        [&](wgpu::CompilationInfoRequestStatus status, wgpu::CompilationInfo const* info) {
            for (size_t i = 0; i < info->messageCount; ++i) {
                const wgpu::CompilationMessage& message = info->messages[i];
                if (message.type == wgpu::CompilationMessageType::Error) {
                    std::cerr << "Shader compilation error: "
                              << std::string(message.message.data, message.message.length)
                              << std::endl;
                } else if (message.type == wgpu::CompilationMessageType::Warning) {
                    std::cerr << "Shader compilation warning: "
                              << std::string(message.message.data, message.message.length)
                              << std::endl;
                }
            }
            promise.set_value();
        });
    std::future<void> future = promise.get_future();
    while (future.wait_for(std::chrono::nanoseconds(100)) == std::future_status::timeout) {
        gpu.instance.ProcessEvents();
    }
    GetComputeShaderPipeline(gpu.device, buffs, cs, entryPoint, mod, csLabel);
}

void GetAllShaders(const GPUContext& gpu, const GPUBuffers& buffs, Shaders* shaders) {
    std::vector<std::string> empty;
    CreateShaderFromSource(gpu, buffs, &shaders->init, "main", "Shaders/init.wgsl", "Init", empty);

    CreateShaderFromSource(gpu, buffs, &shaders->csdldf, "main", "Shaders/csdldf.wgsl", "CSDLDF",
                           empty);

    CreateShaderFromSource(gpu, buffs, &shaders->validate, "main", "Shaders/validate.wgsl",
                           "Validate", empty);
}

void SetComputePass(const ComputeShader& cs, wgpu::CommandEncoder* comEncoder, uint32_t workTiles) {
    wgpu::ComputePassDescriptor comDesc = {};
    comDesc.label = cs.label.c_str();
    wgpu::ComputePassEncoder pass = (*comEncoder).BeginComputePass(&comDesc);
    pass.SetPipeline(cs.computePipeline);
    pass.SetBindGroup(0, cs.bindGroup);
    pass.DispatchWorkgroups(workTiles, 1, 1);
    pass.End();
}

void SetComputePassTimed(const ComputeShader& cs, wgpu::CommandEncoder* comEncoder,
                         const wgpu::QuerySet& querySet, uint32_t workTiles,
                         uint32_t timeStampOffset) {
    wgpu::PassTimestampWrites timeStamp = {};
    timeStamp.beginningOfPassWriteIndex = timeStampOffset * 2;
    timeStamp.endOfPassWriteIndex = timeStampOffset * 2 + 1;
    timeStamp.querySet = querySet;
    wgpu::ComputePassDescriptor comDesc = {};
    comDesc.label = cs.label.c_str();
    comDesc.timestampWrites = &timeStamp;
    wgpu::ComputePassEncoder pass = (*comEncoder).BeginComputePass(&comDesc);
    pass.SetPipeline(cs.computePipeline);
    pass.SetBindGroup(0, cs.bindGroup);
    pass.DispatchWorkgroups(workTiles, 1, 1);
    pass.End();
}

void QueueSync(const GPUContext& gpu) {
    std::promise<void> promise;
    gpu.queue.OnSubmittedWorkDone(wgpu::CallbackMode::AllowProcessEvents,
                                  [&](wgpu::QueueWorkDoneStatus status) {
                                      if (status != wgpu::QueueWorkDoneStatus::Success) {
                                          std::cerr << "uh oh" << std::endl;
                                      }
                                      promise.set_value();
                                  });
    std::future<void> future = promise.get_future();
    while (future.wait_for(std::chrono::nanoseconds(100)) == std::future_status::timeout) {
        gpu.instance.ProcessEvents();
    }
}

void CopyBufferSync(const GPUContext& gpu, wgpu::Buffer* srcReadback, wgpu::Buffer* dstReadback,
                    uint64_t sourceOffsetBytes, uint64_t copySizeBytes) {
    wgpu::CommandEncoderDescriptor comEncDesc = {};
    comEncDesc.label = "Copy Command Encoder";
    wgpu::CommandEncoder comEncoder = gpu.device.CreateCommandEncoder(&comEncDesc);
    comEncoder.CopyBufferToBuffer(*srcReadback, sourceOffsetBytes, *dstReadback, 0ULL,
                                  copySizeBytes);
    wgpu::CommandBuffer comBuffer = comEncoder.Finish();
    gpu.queue.Submit(1, &comBuffer);
    QueueSync(gpu);
}

template <typename T>
void ReadbackSync(const GPUContext& gpu, wgpu::Buffer* dstReadback, std::vector<T>* readOut,
                  uint64_t readbackSizeBytes) {
    std::promise<void> promise;
    dstReadback->MapAsync(
        wgpu::MapMode::Read, 0, readbackSizeBytes, wgpu::CallbackMode::AllowProcessEvents,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView) {
            if (status == wgpu::MapAsyncStatus::Success) {
                const void* data = dstReadback->GetConstMappedRange(0, readbackSizeBytes);
                std::memcpy(readOut->data(), data, readbackSizeBytes);
                dstReadback->Unmap();
            } else {
                std::cerr << "Bad readback" << std::endl;
            }
            promise.set_value();
        });

    std::future<void> future = promise.get_future();
    while (future.wait_for(std::chrono::nanoseconds(100)) == std::future_status::timeout) {
        gpu.instance.ProcessEvents();
    }
}

template <typename T>
void CopyAndReadbackSync(const GPUContext& gpu, wgpu::Buffer* srcReadback,
                         wgpu::Buffer* dstReadback, std::vector<T>* readOut, uint32_t sourceOffset,
                         uint32_t readbackSize) {
    CopyBufferSync(gpu, srcReadback, dstReadback, sourceOffset * sizeof(T),
                   readbackSize * sizeof(T));
    ReadbackSync(gpu, dstReadback, readOut, readbackSize * sizeof(T));
}

bool Validate(const GPUContext& gpu, GPUBuffers* buffs, const ComputeShader& validate) {
    wgpu::CommandEncoderDescriptor comEncDesc = {};
    comEncDesc.label = "Validate Command Encoder";
    wgpu::CommandEncoder comEncoder = gpu.device.CreateCommandEncoder(&comEncDesc);
    SetComputePass(validate, &comEncoder, 256);
    wgpu::CommandBuffer comBuffer = comEncoder.Finish();
    gpu.queue.Submit(1, &comBuffer);
    QueueSync(gpu);

    std::vector<uint32_t> readOut(1, 1);
    CopyAndReadbackSync(gpu, &buffs->misc, &buffs->readback, &readOut, 0, 1);
    bool testPassed = readOut[0] == 0;
    if (!testPassed) {
        std::cerr << "Test failed: " << readOut[0] << " errors" << std::endl;
    }
    return testPassed;
}

void ReadbackAndPrintSync(const GPUContext& gpu, GPUBuffers* buffs, uint32_t readbackSize) {
    std::vector<uint32_t> readOut(readbackSize);
    CopyAndReadbackSync(gpu, &buffs->scanOut, &buffs->readback, &readOut, 0, readbackSize);
    for (uint32_t i = 0; i < (readbackSize + 31) / 32; ++i) {
        for (uint32_t k = 0; k < 32; ++k) {
            std::cout << readOut[i * 32 + k] << ", ";
        }
        std::cout << std::endl;
    }
}

void ResolveTimestampQuerys(GPUBuffers* buffs, const wgpu::QuerySet& query,
                            wgpu::CommandEncoder* comEncoder, uint32_t querySetUnits) {
    (*comEncoder).ResolveQuerySet(query, 0, querySetUnits, buffs->timestamp, 0ULL);
    (*comEncoder)
        .CopyBufferToBuffer(buffs->timestamp, 0ULL, buffs->readbackTimestamp, 0ULL,
                            querySetUnits * sizeof(uint64_t));
}

void InitializeUniforms(const GPUContext& gpu, GPUBuffers* buffs, uint32_t size, uint32_t workTiles,
                        uint32_t simulateMask) {
    wgpu::CommandEncoderDescriptor comEncDesc = {};
    comEncDesc.label = "Initialize Uniforms Command Encoder";
    wgpu::CommandEncoder comEncoder = gpu.device.CreateCommandEncoder(&comEncDesc);
    std::vector<uint32_t> info{size, (size + 3) / 4, workTiles, simulateMask};
    gpu.queue.WriteBuffer(buffs->info, 0ULL, info.data(), info.size() * sizeof(uint32_t));
    wgpu::CommandBuffer comBuffer = comEncoder.Finish();
    gpu.queue.Submit(0, &comBuffer);
    QueueSync(gpu);
}

uint32_t RTS(const TestArgs& args, wgpu::CommandEncoder* comEncoder) {
    const uint32_t passCount = 3;
    if (args.shouldTime) {
        SetComputePassTimed(args.shaders.reduce, comEncoder, args.gpu.querySet,
                            args.workTiles, 0);
        SetComputePassTimed(args.shaders.spineScan, comEncoder,
                            args.gpu.querySet, 1, 1);
        SetComputePassTimed(args.shaders.downsweep, comEncoder,
                            args.gpu.querySet, args.workTiles, 2);
    } else {
        SetComputePass(args.shaders.reduce, comEncoder, args.workTiles);
        SetComputePass(args.shaders.spineScan, comEncoder, 1);
        SetComputePass(args.shaders.downsweep, comEncoder, args.workTiles);
    }
    return passCount;
}

uint32_t CSDL(const TestArgs& args, wgpu::CommandEncoder* comEncoder) {
    const uint32_t passCount = 1;
    if (args.shouldTime) {
        SetComputePassTimed(args.shaders.csdl, comEncoder, args.gpu.querySet,
                            args.workTiles, 0);
    } else {
        SetComputePass(args.shaders.csdl, comEncoder, args.workTiles);
    }
    return passCount;
}

uint32_t CSDLDF(const TestArgs& args, wgpu::CommandEncoder* comEncoder) {
    const uint32_t passCount = 1;
    if (args.shouldTime) {
        SetComputePassTimed(args.shaders.csdldf, comEncoder, args.gpu.querySet,
                            args.workTiles, 0);
    } else {
        SetComputePass(args.shaders.csdldf, comEncoder, args.workTiles);
    }
    return passCount;
}

uint32_t CSDLDF_prof(const TestArgs& args, wgpu::CommandEncoder* comEncoder) {
    const uint32_t passCount = 1;
    if (args.shouldTime) {
        SetComputePassTimed(args.shaders.csdldf, comEncoder, args.gpu.querySet,
                            args.workTiles, 0);
    } else {
        SetComputePass(args.shaders.csdldf, comEncoder, args.workTiles);
    }
    return passCount;
}

uint32_t CSDLDFSimulate(const TestArgs& args,
                        wgpu::CommandEncoder* comEncoder) {
    const uint32_t passCount = 1;
    if (args.shouldTime) {
        SetComputePassTimed(args.shaders.csdldfSimulate, comEncoder,
                            args.gpu.querySet, args.workTiles, 0);
    } else {
        SetComputePass(args.shaders.csdldfSimulate, comEncoder, args.workTiles);
    }
    return passCount;
}

uint32_t Memcpy(const TestArgs& args, wgpu::CommandEncoder* comEncoder) {
    const uint32_t passCount = 1;
    if (args.shouldTime) {
        SetComputePassTimed(args.shaders.memcpy, comEncoder, args.gpu.querySet,
                            args.workTiles, 0);
    } else {
        SetComputePass(args.shaders.memcpy, comEncoder, args.workTiles);
    }

    wgpu::CommandEncoderDescriptor comEncDesc = {};
    comEncDesc.label = "Warmup Command Encoder";
    wgpu::CommandEncoder comEncoder = args.gpu.device.CreateCommandEncoder(&comEncDesc);

    for (uint32_t i = 0; i < args.warmupSize; ++i) {
        SetComputePass(args.shaders.init, &comEncoder, 256);
        SetComputePass(args.shaders.csdldf, &comEncoder, args.workTiles);
    }

    wgpu::CommandBuffer comBuffer = comEncoder.Finish();
    args.gpu.queue.Submit(1, &comBuffer);
    QueueSync(args.gpu);
}

void Run(std::string testLabel, const TestArgs& args) {
    if (args.batchSize == 0) {
        printf("%s: No actual tests to run (batchSize is 0).\n", testLabel.c_str());
        return;
    }

    uint64_t totalTime = 0ULL;
    uint64_t maxTime = 0ULL;
    uint64_t minTime = ~0ULL;
    std::map<uint64_t, unsigned int> timeMap;
    std::multimap<unsigned int, uint64_t> reverseTimeMap;
    std::vector<double> runTimesData(args.shouldRecord ? args.batchSize : 0);

    const uint32_t maxRunsInGPUBatch = args.maxQueryEntries / 2;
    uint32_t numRunsToExecute = args.batchSize;
    uint32_t runsExecutedCount = 0;

    while (runsExecutedCount < numRunsToExecute) {
        uint32_t runsThisGPUBatch =
            std::min(numRunsToExecute - runsExecutedCount, maxRunsInGPUBatch);
        uint32_t queriesThisGPUBatch = runsThisGPUBatch * 2;

        wgpu::CommandEncoderDescriptor comEncDesc = {};
        comEncDesc.label = "Timed Batch Command Encoder";
        wgpu::CommandEncoder comEncoder = args.gpu.device.CreateCommandEncoder(&comEncDesc);

        for (uint32_t j = 0; j < runsThisGPUBatch; ++j) {
            SetComputePass(args.shaders.init, &comEncoder, 256);
            SetComputePassTimed(args.shaders.csdldf, &comEncoder, args.gpu.querySet, args.workTiles,
                                j);
        }
        ResolveTimestampQuerys(&args.buffs, args.gpu.querySet, &comEncoder, queriesThisGPUBatch);

        wgpu::CommandBuffer comBuffer = comEncoder.Finish();
        args.gpu.queue.Submit(1, &comBuffer);
        QueueSync(args.gpu);

        std::vector<uint64_t> batchTimestamps(queriesThisGPUBatch);
        ReadbackSync(args.gpu, &args.buffs.readbackTimestamp, &batchTimestamps,
                     queriesThisGPUBatch * sizeof(uint64_t));

        for (uint32_t k = 0; k < runsThisGPUBatch; ++k) {
            const uint64_t runtime = batchTimestamps[k * 2 + 1] - batchTimestamps[k * 2];
            totalTime += runtime;
            maxTime = std::max(runtime, maxTime);
            minTime = std::min(runtime, minTime);
            timeMap[runtime]++;

            if (args.shouldRecord && !runTimesData.empty()) {
                uint32_t runId = runsExecutedCount + k;
                if (runId < runTimesData.size()) {
                    runTimesData[runId] = static_cast<double>(runtime);
                } else {
                    printf("%s: Internal error during data recording. Run aborted.\n",
                           testLabel.c_str());
                    return;
                }
            }
        }
        runsExecutedCount += runsThisGPUBatch;
    }

    // Quick and dirty single validation past on the final run
    if (!Validate(args.gpu, &args.buffs, args.shaders.validate)) {
        printf("Validation Failed, timing data is suspect, ending early.\n");
        return;
    }

    printf("%s: All %u timed tests complete.\n", testLabel.c_str(), numRunsToExecute);

    double totalTimeAllSec = static_cast<double>(totalTime) / 1e9;

    printf("\n--- Timing Summary for: %s (Timed Runs) ---\n", testLabel.c_str());

    if (numRunsToExecute > 0) {
        printf("Statistics for %u timed test run(s):\n", numRunsToExecute);
        printf("  Total time: %.4f s\n", totalTimeAllSec);

        double avgTimeRunNS = (totalTime > 0 && numRunsToExecute > 0)
                                  ? (static_cast<double>(totalTime) / numRunsToExecute)
                                  : 0.0;
        printf("  Average time per run: %.0f ns\n", avgTimeRunNS);

        if (minTime != ~0ULL) {
            printf("  Min time per run: %llu ns\n", minTime);
        } else {
            printf("  Min time per run: N/A (no tests run or all had issues)\n");
        }
        printf("  Max time per run: %llu ns\n", maxTime);

        for (const auto& pair : timeMap) {
            reverseTimeMap.insert({pair.second, pair.first});
        }
        int topNToPrint = std::min(5, (int)reverseTimeMap.size());
        printf("  Top %d runtimes (runtime ns => number of runs):\n", topNToPrint);
        if (reverseTimeMap.empty()) {
            printf("    { No distinct runtimes recorded }\n");
        } else {
            printf("    { ");
            int numPrinted = 0;
            for (auto it = reverseTimeMap.rbegin();
                 it != reverseTimeMap.rend() && numPrinted < topNToPrint; ++it) {
                if (numPrinted > 0) {
                    printf(", ");
                }
                printf("%llu => %u", it->second, it->first);
                numPrinted++;
            }
            printf(" }\n");
        }

        if (totalTimeAllSec > 0) {
            double throughputElePerSec =
                (static_cast<double>(args.size) * numRunsToExecute) / totalTimeAllSec;
            printf("  Estimated speed (on %u elements/run): %.2e ele/s\n", args.size,
                   throughputElePerSec);
        } else {
            printf("  Estimated speed: N/A\n");
        }
    } else {
        printf("No timed tests were run to analyze.\n");
    }
    printf("---------------------------------------------------\n");

    if (args.shouldRecord && !runTimesData.empty()) {
        std::string csvFilename = testLabel;
        for (char& c : csvFilename) {
            if (c == ' ')
                c = '_';
        }
        RecordToCSV(args, runTimesData, csvFilename);
        printf("Timing data recorded to %s.csv\n", csvFilename.c_str());
    }
}

void TestCSDLDF_occupancy(std::string deviceName, const TestArgs& args) {
    DataStruct data(args);
    GetOccupancySync(args);
    Run(deviceName + "CSDLDF_occupancy", args, CSDLDF_occupancy, data);
    if (args.shouldRecord) {
        RecordToCSV(args, data, deviceName + "CSDLDF_occupancy");
    }
}

static void printUsageAndExit(const char* progName) {
    fprintf(stderr,
            "Usage: %s <deviceName> <size_exponent_N> <warmupSize> <batchSize> "
            "<shouldRecord:true|false|1|0> <toggle_flag>\n",
            progName);
    fprintf(stderr,
            "  <size_exponent_N>: For size = (1 << N). N must be > 10 and < 25 (i.e., 11 <= N <= "
            "24).\n");
    fprintf(stderr, "  <toggle_flag>: Integer bitmask for GPU context toggles 0 <= N < 3.\n");
    exit(EXIT_FAILURE);
}

static bool parseU32(const char* s, uint32_t& val, uint32_t min_incl = 0,
                     uint32_t max_incl = UINT32_MAX) {
    char* end;
    unsigned long n = strtoul(s, &end, 10);
    if (end == s || *end != '\0' || n < min_incl || n > max_incl || n > UINT32_MAX) {
        return false;
    }
    val = static_cast<uint32_t>(n);
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        printUsageAndExit(argv[0]);
    }

    std::string cli_deviceName = argv[1];
    uint32_t cli_size_exponent;
    uint32_t cli_size_actual;
    uint32_t cli_warmupSize;
    uint32_t cli_batchSize;
    bool cli_shouldRecord;
    uint32_t cli_toggle_flag;

    if (!parseU32(argv[2], cli_size_exponent, 11, 25) || !parseU32(argv[3], cli_warmupSize) ||
        !parseU32(argv[4], cli_batchSize) || !parseU32(argv[6], cli_toggle_flag, 0, 3)) {
        printUsageAndExit(argv[0]);
    }
    cli_size_actual = 1U << cli_size_exponent;

    std::string recordStr = argv[5];
    std::transform(recordStr.begin(), recordStr.end(), recordStr.begin(), ::tolower);
    if (recordStr == "true" || recordStr == "1") {
        cli_shouldRecord = true;
    } else if (recordStr == "false" || recordStr == "0") {
        cli_shouldRecord = false;
    } else {
        printUsageAndExit(argv[0]);
    }

    constexpr uint32_t MISC_SIZE = 5;
    constexpr uint32_t PART_SIZE = 4096;
    constexpr uint32_t MAX_READBACK_SIZE = 8192;
    constexpr uint32_t MAX_QUERY_ENTRIES = 4096;
    const uint32_t workTiles = (cli_size_actual + PART_SIZE - 1) / PART_SIZE;

    GPUContext gpu;
    GPUBuffers buffs;
    Shaders shaders;

    GetGPUContext(&gpu, MAX_QUERY_ENTRIES, static_cast<int>(cli_toggle_flag));
    GetGPUBuffers(gpu.device, &buffs, workTiles, MAX_QUERY_ENTRIES, cli_size_actual, MISC_SIZE,
                  MAX_READBACK_SIZE);
    GetAllShaders(gpu, buffs, &shaders);
    InitializeUniforms(gpu, &buffs, cli_size_actual, workTiles, 0);

    TestArgs args = {gpu,
                     buffs,
                     shaders,
                     cli_size_actual,
                     workTiles,
                     cli_warmupSize,
                     cli_batchSize,
                     MAX_QUERY_ENTRIES,
                     cli_shouldRecord};

    RunWarmup(args);
    Run(cli_deviceName + " CSDLDF Timed", args);

    printf("All test phases complete.\n");
    return EXIT_SUCCESS;
}