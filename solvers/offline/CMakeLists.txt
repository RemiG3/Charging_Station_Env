cmake_minimum_required(VERSION 3.5)

project(milp_offline LANGUAGES CXX)

#set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(milp_offline main.cpp)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(GUROBI REQUIRED)

include_directories(${GUROBI_INCLUDE_DIRS})

set(GUROBI_HOME "/home/rgauchot/gurobi1102/linux64")
#set(GRB_LICENSE_FILE "/home/rgauchot/gurobi1102/linux64/gurobi1102/license/gurobi.lic")
include_directories(${GUROBI_HOME}/include)
link_directories(${GUROBI_HOME}/lib)

target_link_libraries(milp_offline
    "${GUROBI_HOME}/lib/libgurobi_c++.a"
    "${GUROBI_HOME}/lib/libgurobi110.so"
    stdc++fs)
