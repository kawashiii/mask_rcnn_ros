FROM osrf/ros:kinetic-desktop-full

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget nano vim gcc make libopencv-dev \
    python3 python3-dev python3-pip && \
    pip3 install --upgrade pip setuptools

RUN mkdir /root/src && cd /root/src && \
    git clone -b lab_config https://github.com/damien-petit/Mask_RCNN &&\
    cd Mask_RCNN && \
    python3 -m pip install -r requirements_cpu.txt && \
    python3 -m pip install pycocotools && \
    python3 setup.py install && \
    wget -q https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 && \
    cd lab && \
    wget -q ftp://dfk:dfk@100.80.196.8/disk1/share/DeepLearning/model/mask_rcnn_lab.h5

RUN jupyter notebook --generate-config && \
    echo 'c.NotebookApp.ip = "*"\nc.NotebookApp.open_browser = False\nc.NotebookApp.token = ""\nc.NotebookApp.allow_root = True' > /root/.jupyter/jupyter_notebook_config.py

RUN apt-get install -y --no-install-recommends \
    python3-tk python-catkin-tools python3-catkin-pkg-modules && \
    python3 -m pip install rospkg catkin_pkg

RUN mkdir -p /root/catkin_ws/src && cd /root/catkin_ws/src && \
    git clone -b lab_config https://github.com/kawashiii/mask_rcnn_ros && \
    mkdir -p /root/catkin_build_ws/src

## after build

# cd /root/catkin_build_ws
# catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -#DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so
# catkin config --install
# cd /root/catkin_build_ws/src
# git clone -b kinetic_lab_config https://github.com/kawashiii/vision_opencv
# cd /root/catkin_build_ws
# catkin build cv_bridge
# source install/setup.bash --extend

# cd /root/catkin_ws/src
# catkin_init_workspace
# cd /root/catkin_ws 
# catkin_make
# cp ~/src/Mask_RCNN/lab/mask_rcnn_lab.h5 ~/catkin_ws/src/mask_rcnn_ros
# source devel/setup.bash
# rosrun mask_rcnn_ros mask_rcnn.py
