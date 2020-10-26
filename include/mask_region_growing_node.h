#ifndef MASK_REGION_GROWING_NODE_H
#define MASK_REGION_GROWING_NODE_H

#include "region_growing_segmentation.h"

//ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/PolygonStamped.h>

#include "mask_rcnn_ros/GetMaskedSurface.h"
#include "mask_rcnn_ros/MaskedObjectAttributes.h"

class MaskRegionGrowingNode {
    public:
        MaskRegionGrowingNode();

	void run();

	void setCameraParameters();
	void callbackDepth(const sensor_msgs::ImageConstPtr& depth_msg);
	bool callbackGetMaskedSurface(mask_rcnn_ros::GetMaskedSurface::Request &req, mask_rcnn_ros::GetMaskedSurface::Response &res);

        void markerInitialization();

    private:
	ros::NodeHandle nh;
	RegionGrowingSegmentation scene_reg;

        sensor_msgs::CameraInfoConstPtr camera_info;
	ros::Subscriber depth_sub;
	ros::ServiceServer get_masked_surface_srv;

	ros::Publisher vis_axes_marker_pub;
	ros::Publisher scene_surface_pointcloud_pub;
	ros::Publisher masked_surface_pointcloud_pub;
	ros::Publisher input_pointcloud_pub;
	ros::Publisher masked_depth_map_pub;

	cv::Mat cameraMatrix;
	cv::Mat distCoeffs;
        cv::Mat map_x, map_y;
	float fx;
	float fy;
	float cx;
	float cy;
	
	cv::Mat depth;
	bool is_service_called;
};

#endif
