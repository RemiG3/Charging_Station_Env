cmake_minimum_required(VERSION 3.5) #3.16

project(Lexico-CPLEX__NEW LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
#set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CPLEX REQUIRED)
add_executable(Lexico-CPLEX__NEW main.cpp)

target_link_libraries(Lexico-CPLEX__NEW cplex-library cplex-concert ilocplex stdc++fs)

#install(TARGETS Test_CPLEX
#    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
