# SuperPointPretrainCpp_TensorFlow
SuperPoint with pretrain model and implement in Tensorflow
This code is built for ROS 
If you want to work without ROS, please delete the ROS portion and edit the CMakeList.
## Model
https://github.com/rpautrat/SuperPoint/tree/master/pretrained_models
## Dependence (with ROS)
```
git clone https://github.com/catkin/catkin_simple.git
git clone https://github.com/ethz-asl/tensorflow_catkin.git
git clone https://github.com/tradr-project/tensorflow_ros_cpp.git
```
put those three folder in the src in the catkin workspace
And then 
```
source /opt/ros/kinetic/setup.bash
catkin build
```
### Without ROS
Change the CmakeList to link with tensorflow.<br />
The origin code was tested in tensorflow ==1.8



 


   
