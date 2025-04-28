cmake_minimum_required(VERSION 3.14)
project( RectFlowStippling)

find_package( OpenCV REQUIRED )
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

include(openmp)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(brush brush.cpp ../flow.cpp)
target_link_libraries( brush ${OpenCV_LIBS}  OpenMP::OpenMP_CXX)


add_executable(webcam webcam.cpp ../flow.cpp)
target_link_libraries( webcam ${OpenCV_LIBS}  OpenMP::OpenMP_CXX)
