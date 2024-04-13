cmake_minimum_required(VERSION 3.5) #3.16

project(Normal_CPLEX_NEW_V2 LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
#set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CPLEX REQUIRED)
add_executable(Normal_CPLEX_NEW_V2 main.cpp)

target_link_libraries(Normal_CPLEX_NEW_V2 cplex-library cplex-concert ilocplex stdc++fs)

#install(TARGETS Test_CPLEX
#    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
