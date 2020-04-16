cmake_minimum_required(VERSION 3.10)

get_filename_component(PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)

set(libname covlp)

add_library(${libname} SHARED ovlp.cpp)

set_target_properties(${libname} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PARENT_DIR})

