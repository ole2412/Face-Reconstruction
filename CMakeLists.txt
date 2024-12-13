cmake_minimum_required(VERSION 3.10)

# Set the project name
project(face-reconstruction)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Eigen3 REQUIRED)
find_package(HDF5 REQUIRED)


set(HEADERS
    src/bfm.h
)

set(SOURCES
    src/bfm.cpp
)

# Add the executable
add_executable(main main.cpp ${HEADERS} ${SOURCES})

target_link_libraries(main Eigen3::Eigen)
target_link_libraries(main HDF5::HDF5)