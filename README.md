# ROS Package of Mask R-CNN
This is a ROS package of Mask-RCNN for object detection and instance segmentation.  
Most of core algorithm code is based on [Mask R-CNN implementation by Matterport, Inc. ](https://github.com/matterport/Mask_RCNN)

## Quickly Start
```
## launch pylon & phoxi node
$ roslaunch pylon_camera pylon_camera_node.launch
$ roslaunch phoxi_camera phoxi_camera.launch

## start and come in docker container, then launch mask-rcnn node
$ sudo docker start mrcnn_ros
$ sudo docker exec -it mrcnn_ros /bin/bash
$ roslaunch mask_rcnn_ros mask_rcnn.launch

## set detection model
$ rosservice call /mask_rcnn_ros/set_model choice

## get depth
$ rosservice call /phoxi_camera/get_frame -- -1

## get detection result
$ rossevice call /mask_rcnn/MaskRCNNSrv
```

## Installation
**1. Docker Build**  
Copy Dockerfile for [cpu](./docker/Dockerfile_cpu) or [gpu](./docker/Dockerfile_gpu) to your folder.
```bash
ln -s Dockerfile_gpu Dockerfile
chmod +x Dockerfile
```
After the copy, build Dockerfile.  
```
$ sudo docker build -t mrcnn_ros:dev .
```

**2. Docker Run**  
```
# cpu
$ sudo docker run --privileged --net=host -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --name mrcnn_ros mrcnn_ros:dev
```  
```
# gpu
$ sudo docker run --privileged --net=host -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --gpus all --name mrcnn_ros mrcnn_ros:dev
```

**3. Installation (Inside Docker Container)**
```
apt install python-pip software-properties-common ros-kinetic-ros-numpy
python -m pip install --upgrade pip
python -m pip install open3d scipy

add-apt-repository ppa:ubuntu-toolchain-r/test
apt update -y
apt upgrade -y
```

**4. catkin init (Inside Docker Container)**
```
prepare the ros environment with the .bashrc file by the following command lines

source /opt/ros/kinetic/setup.bash
source $HOME/catkin_ws/devel/setup.bash
source $HOME/catkin_build_ws/install/setup.bash --extend
export ROS_IP=100.80.196.244


cd /root/catkin_ws/src
catkin_init_workspace
cd /root/catkin_ws 
catkin_make
source devel/setup.bash
```

**5. Build cv_bridge for python3 (Inside Docker Cotainer)**
```
cd /root/catkin_build_ws
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so
catkin config --install
cd /root/catkin_build_ws/src
git clone -b kinetic_lab_config https://github.com/kawashiii/vision_opencv
cd /root/catkin_build_ws
catkin build cv_bridge
source install/setup.bash --extend
```
