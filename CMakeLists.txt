cmake_minimum_required(VERSION 3.23)
project(MNIST)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH ".../Qt6.5.1/6.5.1/macos/lib/cmake")

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(MNIST main.cpp)

target_link_libraries( MNIST ${OpenCV_LIBS} )