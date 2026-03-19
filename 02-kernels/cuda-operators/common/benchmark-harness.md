# Benchmark Harness Template

Use this snippet as the starting point for a concrete operator benchmark file.

```cuda
#include <cuda_runtime.h>

#include <cstdio>

struct BenchmarkResult {
    float milliseconds;
    int iterations;
};

template <typename LaunchFn>
BenchmarkResult benchmark_kernel(LaunchFn launch, int warmup, int iterations) {
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        launch();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return {milliseconds / iterations, iterations};
}
```

## Required Reporting

- GPU model
- input shape
- dtype
- warmup count
- measured iterations
- average time
- effective bandwidth or FLOPs
- baseline versus optimized speedup
