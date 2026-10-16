#include "sensor_simulator_cuda.cuh"
#include "sensor_simulator.h"
#include <cuda_runtime.h>

// CUDA kernel implementations.
template <typename scalar_t>
__global__ void lidarRaycastKernel(
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> lidar_points,   // Six channels: x, y, z, depth, zenith angle, and azimuth angle.
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> balls,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> voxels,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> R,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> pos,
    LidarParams lidar_param) {
    
    // Ray-cast the LiDAR point cloud.
    // Intersect each ray with obstacles and update the lidar_points tensor.
    const int c = blockIdx.x * blockDim.x + threadIdx.x;  // Global linear index.
    const int B = lidar_points.size(0);   // Batch size.
    const int H = lidar_points.size(1);   // Point-cloud height.
    const int W = lidar_points.size(2);   // Point-cloud width.
    if (c >= B * H * W) return;           // Bounds check.
    
    // Convert the linear index to batch, row, and column indices.
    const int b = c / (H * W);       // Batch index.
    const int v = (c % (H * W)) / W; // Row index (vertical).
    const int h = c % W;             // Column index (horizontal).

    // Compute the ray direction.
    float vertical_resolution = (lidar_param.vertical_angle_end - lidar_param.vertical_angle_start) / (lidar_param.vertical_lines - 1);
    float vertical_angle = lidar_param.vertical_angle_start + (lidar_param.vertical_lines - 1 - v) * vertical_resolution;
    float sin_vert = std::sin(vertical_angle * M_PI / 180.0);
    float cos_vert = std::cos(vertical_angle * M_PI / 180.0);
    float horizontal_angle = h * lidar_param.horizontal_resolution;
    float sin_horz = std::sin(horizontal_angle * M_PI / 180.0);
    float cos_horz = std::cos(horizontal_angle * M_PI / 180.0);
    // Compute the ray direction.
    Vector3f ray_direction(cos_vert * cos_horz, cos_vert * sin_horz, sin_vert);  //Convert spherical coordinates to Cartesian coordinates.
    float3 rotated_ray_dir = make_float3(
        R[b][0][0] * ray_direction.x + R[b][0][1] * ray_direction.y + R[b][0][2] * ray_direction.z,
        R[b][1][0] * ray_direction.x + R[b][1][1] * ray_direction.y + R[b][1][2] * ray_direction.z,
        R[b][2][0] * ray_direction.x + R[b][2][1] * ray_direction.y + R[b][2][2] * ray_direction.z);  // Rotated ray direction in the world frame.
    // Extract the ray direction components for each axis.
    float dx = rotated_ray_dir.x;
    float dy = rotated_ray_dir.y;
    float dz = rotated_ray_dir.z;

    // LiDAR position.
    const scalar_t ox = pos[b][0];  
    const scalar_t oy = pos[b][1];
    const scalar_t oz = pos[b][2];

    // Initialize the minimum distance using the ground plane.
    scalar_t min_dist = lidar_param.max_lidar_dist;
    if (fabs(dz) > 1e-6) { // Avoid division by zero.
        scalar_t  t = (0 - oz) / dz;  // Ground plane at z = 0.
        if (t > 0 && t < min_dist) min_dist = t;
    }

    // balls
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[b][i][0];
        scalar_t cy = balls[b][i][1];
        scalar_t cz = balls[b][i][2];
        scalar_t r = balls[b][i][3];
        scalar_t a = dx * dx + dy * dy + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // cylinders
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[b][i][0];
        scalar_t cy = cylinders[b][i][1];
        scalar_t r = cylinders[b][i][2];
        scalar_t a = dx * dx + dy * dy;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[b][i][0];
        scalar_t cz = cylinders_h[b][i][1];
        scalar_t r = cylinders_h[b][i][2];
        scalar_t a = dx * dx + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // voxels
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[b][i][0];
        scalar_t cy = voxels[b][i][1];
        scalar_t cz = voxels[b][i][2];
        scalar_t rx = voxels[b][i][3];
        scalar_t ry = voxels[b][i][4];
        scalar_t rz = voxels[b][i][5];
        scalar_t tx1 = (cx - rx - ox) / dx;
        scalar_t tx2 = (cx + rx - ox) / dx;
        scalar_t tx_min = min(tx1, tx2);
        scalar_t tx_max = max(tx1, tx2);
        scalar_t ty1 = (cy - ry - oy) / dy;
        scalar_t ty2 = (cy + ry - oy) / dy;
        scalar_t ty_min = min(ty1, ty2);
        scalar_t ty_max = max(ty1, ty2);
        scalar_t tz1 = (cz - rz - oz) / dz;
        scalar_t tz2 = (cz + rz - oz) / dz;
        scalar_t tz_min = min(tz1, tz2);
        scalar_t tz_max = max(tz1, tz2);
        scalar_t t_min = max(max(tx_min, ty_min), tz_min);
        scalar_t t_max = min(min(tx_max, ty_max), tz_max);
        if (t_min < min_dist && t_min < t_max && t_min > 0)
            min_dist = t_min;
    }

    lidar_points[b][v][h][0] = min_dist * ray_direction.x;  // Relative position.
    lidar_points[b][v][h][1] = min_dist * ray_direction.y;
    lidar_points[b][v][h][2] = min_dist * ray_direction.z;
    lidar_points[b][v][h][3] = min_dist;       // Range.
    lidar_points[b][v][h][4] = 90.0 - vertical_angle; // Zenith angle.
    lidar_points[b][v][h][5] = horizontal_angle; // Azimuth angle.
    
}

template <typename scalar_t>
__global__ void lidarRaycastKernel_nonre(
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> lidar_points,   // Six channels: x, y, z, depth, zenith angle, and azimuth angle.
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> balls,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> voxels,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> R,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> pos,
    LidarParams lidar_param,
    float start_angle,
    int inc_or_dec // 1 increases the scan angle; -1 decreases it.
    ) {

    // Ray-cast the non-repetitive LiDAR point cloud.
    // Intersect each ray with obstacles and update the lidar_points tensor.
    const int c = blockIdx.x * blockDim.x + threadIdx.x;  // Global linear index.
    const int B = lidar_points.size(0);   // Batch size.
    const int H = lidar_points.size(1);   // Point-cloud height.
    const int W = lidar_points.size(2);   // Point-cloud width.
    if (c >= B * H * W) return;           // Bounds check.
    
    // Convert the linear index to batch, row, and column indices.
    const int b = c / (H * W);       // Batch index.
    const int v = (c % (H * W)) / W; // Row index (vertical).
    const int h = c % W;             // Column index (horizontal).

    // Compute the ray direction.
    float vertical_resolution = lidar_param.vertical_scan_area / (lidar_param.vertical_lines_nonre - 1);
    float vertical_angle = start_angle + (lidar_param.vertical_lines_nonre - 1 - v) * vertical_resolution + inc_or_dec * h * (lidar_param.vertical_angle_increment / (lidar_param.horizontal_num - 1));
    
    float max_vertical_angle = start_angle + (lidar_param.vertical_lines_nonre - 1) * vertical_resolution + inc_or_dec * h * (lidar_param.vertical_angle_increment / (lidar_param.horizontal_num - 1));  // Maximum vertical angle at the current horizontal angle.
    float min_vertical_angle = start_angle + inc_or_dec * h * (lidar_param.vertical_angle_increment / (lidar_param.horizontal_num - 1));  // Minimum vertical angle at the current horizontal angle.
    if (max_vertical_angle > lidar_param.vertical_angle_end){
        // vertical_angle = start_angle + v * vertical_resolution - 1 * h * (lidar_param.vertical_angle_increment / (lidar_param.horizontal_num - 1));
        vertical_angle = -1 * (max_vertical_angle - lidar_param.vertical_angle_end) + vertical_angle;
    }
    if (min_vertical_angle < lidar_param.vertical_angle_start){
        // vertical_angle = start_angle + v * vertical_resolution + 1 * h * (lidar_param.vertical_angle_increment / (lidar_param.horizontal_num - 1));
        vertical_angle = -1 * (min_vertical_angle - lidar_param.vertical_angle_start) + vertical_angle;
    }
    float sin_vert = std::sin(vertical_angle * M_PI / 180.0);
    float cos_vert = std::cos(vertical_angle * M_PI / 180.0);
    float horizontal_angle = h * lidar_param.horizontal_resolution;
    float sin_horz = std::sin(horizontal_angle * M_PI / 180.0);
    float cos_horz = std::cos(horizontal_angle * M_PI / 180.0);
    // Compute the ray direction.
    Vector3f ray_direction(cos_vert * cos_horz, cos_vert * sin_horz, sin_vert);  //Convert spherical coordinates to Cartesian coordinates.
    float3 rotated_ray_dir = make_float3(
        R[b][0][0] * ray_direction.x + R[b][0][1] * ray_direction.y + R[b][0][2] * ray_direction.z,
        R[b][1][0] * ray_direction.x + R[b][1][1] * ray_direction.y + R[b][1][2] * ray_direction.z,
        R[b][2][0] * ray_direction.x + R[b][2][1] * ray_direction.y + R[b][2][2] * ray_direction.z);  // Rotated ray direction in the world frame.
    // Extract the ray direction components for each axis.
    float dx = rotated_ray_dir.x;
    float dy = rotated_ray_dir.y;
    float dz = rotated_ray_dir.z;

    // LiDAR position.
    const scalar_t ox = pos[b][0];  
    const scalar_t oy = pos[b][1];
    const scalar_t oz = pos[b][2];

    // Initialize the minimum distance using the ground plane.
    scalar_t min_dist = lidar_param.max_lidar_dist;
    if (fabs(dz) > 1e-6) { // Avoid division by zero.
        scalar_t  t = (0 - oz) / dz;  // Ground plane at z = 0.
        if (t > 0 && t < min_dist) min_dist = t;
    }

    // balls
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[b][i][0];
        scalar_t cy = balls[b][i][1];
        scalar_t cz = balls[b][i][2];
        scalar_t r = balls[b][i][3];
        scalar_t a = dx * dx + dy * dy + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // cylinders
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[b][i][0];
        scalar_t cy = cylinders[b][i][1];
        scalar_t r = cylinders[b][i][2];
        scalar_t a = dx * dx + dy * dy;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[b][i][0];
        scalar_t cz = cylinders_h[b][i][1];
        scalar_t r = cylinders_h[b][i][2];
        scalar_t a = dx * dx + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // voxels
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[b][i][0];
        scalar_t cy = voxels[b][i][1];
        scalar_t cz = voxels[b][i][2];
        scalar_t rx = voxels[b][i][3];
        scalar_t ry = voxels[b][i][4];
        scalar_t rz = voxels[b][i][5];
        scalar_t tx1 = (cx - rx - ox) / dx;
        scalar_t tx2 = (cx + rx - ox) / dx;
        scalar_t tx_min = min(tx1, tx2);
        scalar_t tx_max = max(tx1, tx2);
        scalar_t ty1 = (cy - ry - oy) / dy;
        scalar_t ty2 = (cy + ry - oy) / dy;
        scalar_t ty_min = min(ty1, ty2);
        scalar_t ty_max = max(ty1, ty2);
        scalar_t tz1 = (cz - rz - oz) / dz;
        scalar_t tz2 = (cz + rz - oz) / dz;
        scalar_t tz_min = min(tz1, tz2);
        scalar_t tz_max = max(tz1, tz2);
        scalar_t t_min = max(max(tx_min, ty_min), tz_min);
        scalar_t t_max = min(min(tx_max, ty_max), tz_max);
        if (t_min < min_dist && t_min < t_max && t_min > 0)
            min_dist = t_min;
    }

    lidar_points[b][v][h][0] = min_dist * ray_direction.x;  // Relative position.
    lidar_points[b][v][h][1] = min_dist * ray_direction.y;
    lidar_points[b][v][h][2] = min_dist * ray_direction.z;
    lidar_points[b][v][h][3] = min_dist;       // Range.
    lidar_points[b][v][h][4] = 90.0 - vertical_angle; // Zenith angle.
    lidar_points[b][v][h][5] = horizontal_angle; // Azimuth angle.
}

template <typename scalar_t>
__global__ void depthImageKernel(
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> canvas,   // Single-channel depth data.
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> balls,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> cylinders,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> cylinders_h,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> voxels,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> R,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> pos,
    float fov_x_half_tan) {
    
    // Ray-cast the depth image.
    // Intersect each pixel ray with obstacles and update the canvas tensor.
    const int c = blockIdx.x * blockDim.x + threadIdx.x;  // Global linear index.
    const int B = canvas.size(0);   // Batch size.
    const int H = canvas.size(1);   // Image height.
    const int W = canvas.size(2);   // Image width.
    if (c >= B * H * W) return;     // Bounds check.

    // Convert the linear index to batch, row, and column indices.
    const int b = c / (H * W);       // Batch index.
    const int v = (c % (H * W)) / W; // Row index (vertical).
    const int h = c % W;             // Column index (horizontal).

    // Compute camera projection parameters for ray casting.
    const scalar_t fov_y_half_tan = fov_x_half_tan / W * H;
    const scalar_t fv = (2 * (v + 0.5) / H - 1) * fov_y_half_tan - 1e-5;   // Convert pixel coordinates to camera-space directions; sample pixel centers and normalize the vertical coordinate to [-1, 1).
    const scalar_t fh = (2 * (h + 0.5) / W - 1) * fov_x_half_tan - 1e-5;   // Scale normalized coordinates by the half-FOV tangent; subtract a small epsilon to avoid division-by-zero or boundary issues.
    
    // Compute the ray direction in the world frame using rotation matrix R.
    scalar_t dx = R[b][0][0] - fv * R[b][0][2] - fh * R[b][0][1];
    scalar_t dy = R[b][1][0] - fv * R[b][1][2] - fh * R[b][1][1];
    scalar_t dz = R[b][2][0] - fv * R[b][2][2] - fh * R[b][2][1];
    const scalar_t ox = pos[b][0];  // Camera position.
    const scalar_t oy = pos[b][1];
    const scalar_t oz = pos[b][2];
    
    // Compute the minimum distance from each pixel ray to nearby obstacles.
    // Initialize the minimum distance using the ground plane.
    scalar_t min_dist = 100;
    scalar_t  t = (0 - oz) / dz;  // Ground plane at z = 0.
    if (t > 0) min_dist = t;

    // balls
    for (int i = 0; i < balls.size(1); i++) {
        scalar_t cx = balls[b][i][0];
        scalar_t cy = balls[b][i][1];
        scalar_t cz = balls[b][i][2];
        scalar_t r = balls[b][i][3];
        scalar_t a = dx * dx + dy * dy + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // cylinders
    for (int i = 0; i < cylinders.size(1); i++) {
        scalar_t cx = cylinders[b][i][0];
        scalar_t cy = cylinders[b][i][1];
        scalar_t r = cylinders[b][i][2];
        scalar_t a = dx * dx + dy * dy;
        scalar_t b = 2 * (dx * (ox - cx) + dy * (oy - cy));
        scalar_t c = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }
    for (int i = 0; i < cylinders_h.size(1); i++) {
        scalar_t cx = cylinders_h[b][i][0];
        scalar_t cz = cylinders_h[b][i][1];
        scalar_t r = cylinders_h[b][i][2];
        scalar_t a = dx * dx + dz * dz;
        scalar_t b = 2 * (dx * (ox - cx) + dz * (oz - cz));
        scalar_t c = (ox - cx) * (ox - cx) + (oz - cz) * (oz - cz) - r * r;
        scalar_t d = b * b - 4 * a * c;
        if (d >= 0) {
            r = (-b-sqrt(d)) / (2 * a);
            if (r > 1e-5) {
                min_dist = min(min_dist, r);
            } else {
                r = (-b+sqrt(d)) / (2 * a);
                if (r > 1e-5) min_dist = min(min_dist, r);
            }
        }
    }

    // voxels
    for (int i = 0; i < voxels.size(1); i++) {
        scalar_t cx = voxels[b][i][0];
        scalar_t cy = voxels[b][i][1];
        scalar_t cz = voxels[b][i][2];
        scalar_t rx = voxels[b][i][3];
        scalar_t ry = voxels[b][i][4];
        scalar_t rz = voxels[b][i][5];
        scalar_t tx1 = (cx - rx - ox) / dx;
        scalar_t tx2 = (cx + rx - ox) / dx;
        scalar_t tx_min = min(tx1, tx2);
        scalar_t tx_max = max(tx1, tx2);
        scalar_t ty1 = (cy - ry - oy) / dy;
        scalar_t ty2 = (cy + ry - oy) / dy;
        scalar_t ty_min = min(ty1, ty2);
        scalar_t ty_max = max(ty1, ty2);
        scalar_t tz1 = (cz - rz - oz) / dz;
        scalar_t tz2 = (cz + rz - oz) / dz;
        scalar_t tz_min = min(tz1, tz2);
        scalar_t tz_max = max(tz1, tz2);
        scalar_t t_min = max(max(tx_min, ty_min), tz_min);
        scalar_t t_max = min(min(tx_max, ty_max), tz_max);
        if (t_min < min_dist && t_min < t_max && t_min > 0)
            min_dist = t_min;
    }

    canvas[b][v][h] = min_dist;
}

// Host wrapper implementations.
void renderLidarPointcloud(
    torch::Tensor lidar_points,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor pos,
    LidarParams lidar_param){
    
    const int threads = lidar_param.horizontal_num;                   // Use horizontal_num threads per block.
    size_t state_size = lidar_points.size(0) * lidar_points.size(1) * lidar_points.size(2);                         // Number of output samples (height x width x batch size).
    const dim3 blocks((state_size + threads - 1) / threads);          // Compute the block count to cover all output samples.

    AT_DISPATCH_FLOATING_TYPES(lidar_points.type(), "lidarRaycastKernel", ([&] {
        lidarRaycastKernel<scalar_t><<<blocks, threads>>>(
            lidar_points.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),  // Six channels: x, y, z, depth, zenith angle, and azimuth angle.
            balls.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            cylinders.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            cylinders_h.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            voxels.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            R.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            pos.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            lidar_param);
    }));
    
}

void renderLidarPointcloud_nonre(
    torch::Tensor lidar_points,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor pos,
    LidarParams lidar_param,
    float start_angle,
    int inc_or_dec // 1 increases the scan angle; -1 decreases it.
    ){
    
    const int threads = lidar_param.horizontal_num;                   // Use horizontal_num threads per block.
    size_t state_size = lidar_points.size(0) * lidar_points.size(1) * lidar_points.size(2);                         // Number of output samples (height x width x batch size).
    const dim3 blocks((state_size + threads - 1) / threads);          // Compute the block count to cover all output samples.

    AT_DISPATCH_FLOATING_TYPES(lidar_points.type(), "lidarRaycastKernel", ([&] {
        lidarRaycastKernel_nonre<scalar_t><<<blocks, threads>>>(
            lidar_points.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),   // Six channels: x, y, z, depth, zenith angle, and azimuth angle.
            balls.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            cylinders.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            cylinders_h.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            voxels.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            R.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            pos.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            lidar_param,
            start_angle,
            inc_or_dec);
    }));
    
}

void renderDepthImage(
    torch::Tensor canvas,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor pos,
    float fov_x_half_tan){
    
    const int threads = canvas.size(2);                   // Use one thread per canvas column in each block.
    size_t state_size = canvas.size(0) * canvas.size(1) * canvas.size(2);  // Number of output samples (height x width x batch size).
    const dim3 blocks((state_size + threads - 1) / threads);          //    Compute the block count to cover all output samples.

    AT_DISPATCH_FLOATING_TYPES(canvas.type(), "depthImageKernel", ([&] {
        depthImageKernel<scalar_t><<<blocks, threads>>>(
            canvas.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),   // Single-channel depth data.
            balls.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            cylinders.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            cylinders_h.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            voxels.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            R.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            pos.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            fov_x_half_tan);
    }));
}

