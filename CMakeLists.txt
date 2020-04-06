cmake_minimum_required(VERSION 2.8.3)
project(tensorflow_superpoint)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  tensorflow_ros_cpp
)
find_package( OpenCV REQUIRED )

catkin_package()

include_directories(
  # needs to be first so that the tensorflow versions of includes are always treated first (not needed on all systems, but sometimes it is).
  ${tensorflow_ros_cpp_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(tensorflow_superpoint_node src/superpoint.cpp)


# again, tensorflow_ros_cpp_LIBRARIES need to go first in some setups

target_link_libraries(tensorflow_superpoint_node ${tensorflow_ros_cpp_LIBRARIES} ${catkin_LIBRARIES})
target_link_libraries(tensorflow_superpoint_node ${OpenCV_LIBS})



