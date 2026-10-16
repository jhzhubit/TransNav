#ifndef SENSOR_SIMULATOR_CUDA_CUH
#define SENSOR_SIMULATOR_CUDA_CUH

#include <torch/extension.h>

struct Vector3f
    {
        float x, y, z;
        __device__ __host__ Vector3f() : x(0.0f), y(0.0f), z(0.0f) {}
        __device__ __host__ Vector3f(float x_val, float y_val, float z_val)
            : x(x_val), y(y_val), z(z_val) {}
    };

    struct Vector3i
    {
        int x, y, z;
        __device__ __host__ Vector3i() : x(0), y(0), z(0) {}
        __device__ __host__ Vector3i(int x_val, int y_val, int z_val)
            : x(x_val), y(y_val), z(z_val) {}
    };

#endif // SENSOR_SIMULATOR_CUDA_CUH