cmake_minimum_required(VERSION 3.22.1)

project(yolo26ncnn)

# 设置 ncnn 路径 - 请根据实际解压路径修改
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)

find_package(ncnn REQUIRED)

# 添加共享库
add_library(yolo26ncnn SHARED
    yolo26ncnn.cpp
    yolo.cpp
)

# 链接库
target_link_libraries(yolo26ncnn
    ncnn
    mediandk
    android
    log
    jnigraphics
)
