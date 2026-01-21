cmake_minimum_required(VERSION 3.5)

project(lib_online_rolling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(GUROBI REQUIRED)

include_directories(${GUROBI_INCLUDE_DIRS})

set(GUROBI_HOME "~/gurobi1102/linux64")
include_directories(${GUROBI_HOME}/include)
link_directories(${GUROBI_HOME}/lib)

add_library(lib_online_rolling SHARED
            online_gurobi.cpp
            online_gurobi.h
)

target_link_libraries(lib_online_rolling
    "${GUROBI_HOME}/lib/libgurobi_c++.a"
    "${GUROBI_HOME}/lib/libgurobi110.so"
    stdc++fs)

target_compile_definitions(lib_online_rolling PRIVATE LIB_ONLINE_ROLLING_LIBRARY)

