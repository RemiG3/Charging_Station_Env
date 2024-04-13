cmake_minimum_required(VERSION 3.5)

project(lib_online_rolling LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CPLEX REQUIRED)

add_library(lib_online_rolling SHARED
  online_cplex.cpp
  online_cplex.h
)

target_link_libraries(lib_online_rolling cplex-library cplex-concert ilocplex stdc++fs)

target_compile_definitions(lib_online_rolling PRIVATE LIB_ONLINE_ROLLING_LIBRARY)
