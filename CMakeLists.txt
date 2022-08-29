cmake_minimum_required(VERSION 2.8)
project(mvo)

message(STATUS "${PROJECT_NAME}")
message(STATUS "${PROJECT_SOURCE_DIR}")
message(STATUS "${PROJECT_BINARY_DIR}")


find_package(OpenCV 3.2)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV 3.2 not found.")
   endif()

find_package(OpenGL REQUIRED)
   if(NOT OpenGL_FOUND)
      message(FATAL_ERROR "OpenGL not found.")
   endif()

find_package(GLUT REQUIRED)
   if(NOT GLUT_FOUND)
      message(FATAL_ERROR "GLUT not found.")
   endif()

find_package(Pangolin REQUIRED)
   if(NOT Pangolin_FOUND)
      message(FATAL_ERROR "Pangolin not found.")
   endif()


MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})


include_directories(
   ${EIGEN3_INCLUDE_DIRS}
   ${GLUT_INCLUDE_DIRS}
   ${OpenCV_INCLUDE_DIRS}
   ${OPENGL_INCLUDE_DIRS}
   ${Pangolin_INCLUDE_DIRS}
   ${PROJECT_SOURCE_DIR}/include/
)

add_executable(mvo
   ${PROJECT_SOURCE_DIR}/main/main.cpp
   ${PROJECT_SOURCE_DIR}/main/Init.cpp
)

target_link_libraries(mvo 
		  ${PROJECT_SOURCE_DIR}/src/Feature.cpp
		  ${PROJECT_SOURCE_DIR}/src/PoseEstimation.cpp
)

target_link_libraries(mvo
   ${EIGEN3_LIBS}
   ${GLUT_LIBRARY}
   ${OpenCV_LIBS}
   ${OPENGL_LIBRARIES}
   ${Pangolin_LIBRARIES}
   pango_display
   pango_python
)



target_COMPILE_OPTIONS(mvo PUBLIC -g -Wall)


