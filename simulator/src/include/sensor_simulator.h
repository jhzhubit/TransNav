#ifndef SENSOR_SIMULATOR_H
#define SENSOR_SIMULATOR_H

#include <torch/extension.h>

struct LidarParams
    {
        int vertical_lines = 32;            // 32 vertical scan lines.
        float vertical_angle_start = -7.0;  // Starting vertical angle in degrees.
        float vertical_angle_end = 52.0;    // Ending vertical angle in degrees.
        int horizontal_num = 360;           // 360 horizontal samples.
        float horizontal_resolution = 1.0;  // Horizontal resolution of 1 degree.
        float max_lidar_dist{20};           // Maximum LiDAR detection range.

        int vertical_lines_nonre = 4;         // Number of non-repetitive scan lines.
        float vertical_scan_area = 8.0;       // Vertical angular range covered by non-repetitive scan lines.
        float vertical_angle_increment = 6.7; // Vertical angle increment per non-repetitive scan revolution.

        LidarParams() : 
        vertical_lines(32),    
        vertical_angle_start(-7.0f),    
        vertical_angle_end(52.0f),     
        horizontal_num(360),    
        horizontal_resolution(1.0f),
        max_lidar_dist(20),
        vertical_lines_nonre(4),
        vertical_scan_area(8.0f),
        vertical_angle_increment(6.7f) {}
    };

#endif // SENSOR_SIMULATOR_H
