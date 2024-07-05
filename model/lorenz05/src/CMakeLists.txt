cmake_minimum_required(VERSION 3.16)

enable_language(Fortran)

file(GLOB_RECURSE sources *.f90)
add_library(tools STATIC ${sources})
