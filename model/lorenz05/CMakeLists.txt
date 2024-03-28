cmake_minimum_required(VERSION 3.16)

project(L05 LANGUAGES Fortran)

add_subdirectory(src)

add_executable(lorenz2_test lorenz2_test.f90)
target_link_libraries(lorenz2_test tools)
include_directories(${CMAKE_BINARY_DIR}/src)