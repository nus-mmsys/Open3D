// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <vector>

#include "open3d/core/Device.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Parallel.h"

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>

#include "open3d/core/CUDAState.cuh"
#include "open3d/core/CUDAUtils.h"
#endif

namespace open3d {
namespace core {

static constexpr int64_t OPEN3D_PARFOR_GRAIN = 32767;
static constexpr int64_t OPEN3D_PARFOR_BLOCK = 128;
static constexpr int64_t OPEN3D_PARFOR_THREAD = 4;

namespace kernelnew {

#ifndef __CUDACC__
namespace cpu_launcher {
/// The value is chosen heuristically for small element-wise ops. When the
/// number of workloads is smaller or equal to OPEN3D_PARFOR_GRAIN, the
/// workloads are executed in serial, otherwise they are executed in parallel.

/// \brief Run a function in parallel on CPU.
///
/// This is typically used together with cuda_launcher::ParallelFor() to
/// share the same code between CPU and CUDA. For example:
///
/// ```cpp
/// #if defined(__CUDACC__)
///     namespace launcher = core::kernelnew::cuda_launcher;
/// #else
///     namespace launcher = core::kernelnew::cpu_launcher;
/// #endif
///
/// launcher::ParallelFor(num_workloads, [=] OPEN3D_DEVICE(int64_t idx) {
///     process_workload(idx);
/// });
/// ```
///
/// \param n The number of workloads.
/// \param func The function to be executed in parallel. The function should
/// take an int64_t workload index and returns void, i.e., `void func(int64_t)`.
///
/// \note This is optimized for uniform work items, i.e. where each call to \p
/// func takes the same time.
/// \note If you use a lambda function, capture only the required variables
/// instead of all to prevent accidental race conditions. If you want the
/// kernelnew to be used on both CPU and CUDA, capture the variables by value.
template <typename func_t>
void ParallelFor(int64_t n, const func_t& func) {
#pragma omp parallel for num_threads(utility::EstimateMaxThreads())
    for (int64_t i = 0; i < n; ++i) {
        func(i);
    }
}

/// Run a function in parallel on CPU when the number of workloads is larger
/// than a threshold.
///
/// \param n The number of workloads.
/// \param grain_size If \p n <= \p grain_size, the jobs will be executed in
/// serial.
/// \param func The function to be executed in parallel. The function should
/// take an int64_t workload index and returns void, i.e., `void func(int64_t)`.
template <typename func_t>
void ParallelFor(int64_t n, int64_t grain_size, const func_t& func) {
#pragma omp parallel for schedule(static) if (n > grain_size) \
        num_threads(utility::EstimateMaxThreads())
    for (int64_t i = 0; i < n; ++i) {
        func(i);
    }
}
}  // namespace cpu_launcher

#else
namespace cuda_launcher {

/// Calls f(n) with the "grid-stride loops" pattern.
template <int64_t block_size, int64_t thread_size, typename func_t>
__global__ void ElementWiseKernel(int64_t n, func_t f) {
    int64_t items_per_block = block_size * thread_size;
    int64_t idx = blockIdx.x * items_per_block + threadIdx.x;
#pragma unroll
    for (int64_t i = 0; i < thread_size; i++) {
        if (idx < n) {
            f(idx);
            idx += block_size;
        }
    }
}

/// Run a function in parallel with CUDA.
///
/// This is typically used together with cpu_launcher::ParallelFor() to share
/// the same code between CPU and CUDA. For example:
///
/// ```cpp
/// #if defined(__CUDACC__)
///     namespace launcher = core::kernelnew::cuda_launcher;
/// #else
///     namespace launcher = core::kernelnew::cpu_launcher;
/// #endif
///
/// launcher::ParallelFor(num_workloads, [=] OPEN3D_DEVICE(int64_t idx) {
///     process_workload(idx);
/// });
/// ```
///
/// \param n The number of workloads.
/// \param func The function to be executed in parallel. The function should
/// take an int64_t workload index and returns void, i.e., `void func(int64_t)`.
///
/// \note This is optimized for uniform work items, i.e. where each call to \p
/// func takes the same time.
/// \note If you use a lambda function, capture only the required variables
/// instead of all to prevent accidental race conditions. If you want the
/// kernelnew to be used on both CPU and CUDA, capture the variables by value.
template <typename func_t>
void ParallelFor(int64_t n, const func_t& func) {
    if (n == 0) {
        return;
    }
    int64_t items_per_block = OPEN3D_PARFOR_BLOCK * OPEN3D_PARFOR_THREAD;
    int64_t grid_size = (n + items_per_block - 1) / items_per_block;

    ElementWiseKernel<OPEN3D_PARFOR_BLOCK, OPEN3D_PARFOR_THREAD>
            <<<grid_size, OPEN3D_PARFOR_BLOCK, 0, core::cuda::GetStream()>>>(
                    n, func);
    OPEN3D_GET_LAST_CUDA_ERROR("ParallelFor failed.");
}

}  // namespace cuda_launcher
#endif

}  // namespace kernelnew

template <typename func_t>
void ParallelFor(const Device& device, int64_t n, const func_t& func) {
    if (device.GetType() == Device::DeviceType::CPU) {
#ifndef __CUDACC__
        kernelnew::cpu_launcher::ParallelFor(n, func);
#else
        utility::LogError(
                "ParallelFor cannot run on {} with NVCC compiled code.",
                device.ToString());
#endif
    } else if (device.GetType() == Device::DeviceType::CUDA) {
#ifndef __CUDACC__
        utility::LogError(
                "ParallelFor cannot run on {} with non-NVCC compiled code.",
                device.ToString());
#else
        CUDAScopedDevice scoped_device(device);
        kernelnew::cuda_launcher::ParallelFor(n, func);
#endif
    }
}

#ifndef __CUDACC__
template <typename func_t>
void ParallelForGrained(int64_t n,
                        const func_t& func,
                        int64_t grain_size = OPEN3D_PARFOR_GRAIN) {
    kernelnew::cpu_launcher::ParallelFor(n, grain_size, func);
}
#endif

}  // namespace core
}  // namespace open3d
