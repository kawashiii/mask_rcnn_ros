# ROS Package of Mask R-CNN
This is a ROS package of Mask-RCNN for object detection and instance segmentation.  
Most of core algorithm code is based on [Mask R-CNN implementation by Matterport, Inc. ](https://github.com/matterport/Mask_RCNN)

## Requirements
- ROS kinetic

## Installation
**1. Docker Build**  
Copy docker/Dockerfile_cpu(gpu) - select one to suit your environment - to {ANY_FOLDER}/Dockerfile.  
After the copy, build Dockerfile.  
```
sudo docker build -t mrcnn_ros:dev .
```
**2. Docker Run**  
```
# cpu
sudo docker run --privileged --net=host -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --name mrcnn_ros mrcnn_ros:dev
```  
```
# gpu
sudo docker run --privileged --net=host -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --gpus all --name mrcnn_ros mrcnn_ros:dev
```

**3. Setting Catkin**
```
cd /root/catkin_build_ws
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so
catkin config --install
cd /root/catkin_build_ws/src
git clone -b kinetic_lab_config https://github.com/kawashiii/vision_opencv
cd /root/catkin_build_ws
catkin build cv_bridge
source install/setup.bash --extend

cd /root/catkin_ws/src
catkin_init_workspace
cd /root/catkin_ws 
catkin_make
cp ~/src/Mask_RCNN/lab/mask_rcnn_lab.h5 ~/catkin_ws/src/mask_rcnn_ros
source devel/setup.bash
```

## Getting Started
```
rosrun mask_rcnn_ros mask_rcnn.py
```

## ROS Interfaces
### Topics Published
- /mask_rcnn/MaskRCNNMsg: MaskRCNNMsg
- /mask_rcnn/visualization: sensor_msg/Image
### Service
- /mask_rcnn/MaskRCNNSrv

