cmake_minimum_required(VERSION 3.10)

# Set the project name
project(face-reconstruction)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Set the HDF5 include and library paths based on USE_MPI
option(USE_MPI "Use HDF5 with MPI support" OFF)

if(USE_MPI)
    set(HDF5_INCLUDE_DIRS /usr/include/hdf5/openmpi)
    set(HDF5_LIBRARIES /usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5.so /usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5_cpp.so)
    message(STATUS "Using HDF5 with MPI support")
else()
    set(HDF5_INCLUDE_DIRS /usr/include/hdf5/serial)
    set(HDF5_LIBRARIES /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_cpp.so)
    message(STATUS "Using HDF5 without MPI support")
endif()

# Include directories for HDF5 and Eigen
include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

# Set headers and sources
set(HEADERS
    src/bfm.h
    src/render.h
)

set(SOURCES
    src/bfm.cpp
    src/render.cpp
)

# Add the main executable target
add_executable(main main.cpp ${HEADERS} ${SOURCES})

# Link Eigen and HDF5 libraries (both C and C++ libraries)
target_link_libraries(main Eigen3::Eigen ${HDF5_LIBRARIES})

# Print paths to verify configuration
message(STATUS "HDF5 include directories: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 libraries: ${HDF5_LIBRARIES}")
message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "Eigen3 include directories: ${EIGEN3_INCLUDE_DIR}")
