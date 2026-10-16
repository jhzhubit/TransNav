#include "sensor_simulator.h"
#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void renderLidarPointcloud(
    torch::Tensor lidar_points,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor pos,
    LidarParams lidar_param);

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
    );

void renderDepthImage(
    torch::Tensor canvas,
    torch::Tensor balls,
    torch::Tensor cylinders,
    torch::Tensor cylinders_h,
    torch::Tensor voxels,
    torch::Tensor R,
    torch::Tensor pos,
    float fov_x_half_tan
  );



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<LidarParams>(m, "LidarParams")  // The second argument is the Python class name.
        // Bind the default constructor to allow calls such as LidarParams().
        .def(py::init<>())
        // Expose struct members as readwrite or readonly as needed.
        .def_readwrite("vertical_lines", &LidarParams::vertical_lines) 
        .def_readwrite("vertical_angle_start", &LidarParams::vertical_angle_start)   
        .def_readwrite("vertical_angle_end", &LidarParams::vertical_angle_end)    
        .def_readwrite("horizontal_num", &LidarParams::horizontal_num)    
        .def_readwrite("horizontal_resolution", &LidarParams::horizontal_resolution)
        .def_readwrite("max_lidar_dist", &LidarParams::max_lidar_dist)
        .def_readwrite("vertical_lines_nonre", &LidarParams::vertical_lines_nonre)
        .def_readwrite("vertical_scan_area", &LidarParams::vertical_scan_area)
        .def_readwrite("vertical_angle_increment", &LidarParams::vertical_angle_increment);   

  m.def("render", &renderLidarPointcloud, "render (CUDA)");
  m.def("render_nonre", &renderLidarPointcloud_nonre, "render (non-repetitive, CUDA)");
  m.def("render_depth", &renderDepthImage, "render depth image (CUDA)");
}
