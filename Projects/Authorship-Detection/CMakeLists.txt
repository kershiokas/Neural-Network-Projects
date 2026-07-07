cmake_minimum_required(VERSION 4.0)
project(authorship_network)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

add_executable(authorship_network main.cpp
        matrix.h
        Layer.h
        Network.h
        Activations.h)

target_compile_options(authorship_network PRIVATE
        -O3
        -march=native
        -funroll-loops
        -ffast-math
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(authorship_network OpenMP::OpenMP_CXX)
endif()
