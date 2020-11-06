#ifndef MASK_REGION_GROWING_NODE_H
#define MASK_REGION_GROWING_NODE_H


//ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <pcl_ros/point_cloud.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/PolygonStamped.h>
#include "region_growing_segmentation.h"

#include "mask_rcnn_ros/GetMaskedSurface.h"
#include "mask_rcnn_ros/MaskedObjectAttributes.h"

class MaskRegionGrowingNode {
    public:
        MaskRegionGrowingNode();

	void run();

	void setCameraParameters();
        void markerInitialization();

	void callbackDepth(const sensor_msgs::ImageConstPtr& depth_msg);
	bool callbackGetMaskedSurface(mask_rcnn_ros::GetMaskedSurface::Request &req, mask_rcnn_ros::GetMaskedSurface::Response &res);

        void maskedRegionGrowing(cv::Mat mask);
	mask_rcnn_ros::MaskedObjectAttributes build_moa_msg(PointCloudT::Ptr cloud, NormalCloudT::Ptr normal_cloud, int center_index, float area, MomentOfInertia moi);

	void publishPointCloud();
	void publishMarkerArray();
	void publishMaskedDepthMap();

	visualization_msgs::Marker buildDelMarker();
	visualization_msgs::Marker buildSphereMarker(std_msgs::Header header, std::string ns, int id, geometry_msgs::Point point, std_msgs::ColorRGBA color);
	visualization_msgs::Marker buildSphereListMarker(std_msgs::Header header, std::string ns, int id, std::vector<geometry_msgs::Point> points, std_msgs::ColorRGBA color);
	visualization_msgs::Marker buildArrowMarker(std_msgs::Header header, std::string ns, int id, geometry_msgs::Point point, geometry_msgs::Vector3 vector, std_msgs::ColorRGBA color);
	visualization_msgs::Marker buildTextMarker(std_msgs::Header header, std::string ns, int id, geometry_msgs::Point point, std::string text, std_msgs::ColorRGBA color);

    private:
	ros::NodeHandle nh;
	float timeout;
	RegionGrowingSegmentation scene_reg;

	std::string frame_id;
        sensor_msgs::CameraInfoConstPtr camera_info;
	ros::Subscriber depth_sub;
	ros::ServiceServer get_masked_surface_srv;

	ros::Publisher markers_pub;
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
	
	std::vector<sensor_msgs::Image> mask_msgs;
	cv::Mat depth;
	bool is_service_called;

	std_msgs::ColorRGBA x_axis_color;
	std_msgs::ColorRGBA y_axis_color;
	std_msgs::ColorRGBA z_axis_color;
	std_msgs::ColorRGBA polygon_color;
	std_msgs::ColorRGBA text_color;
	geometry_msgs::Vector3 arrow_scale;
	geometry_msgs::Vector3 sphere_scale;
	geometry_msgs::Vector3 text_scale;

	PointCloudT::Ptr input_scene;
	std::vector<PointCloudColorT::Ptr> scene_surface_list;
	std::vector<PointCloudColorT::Ptr> masked_surface_list;
	std::vector<mask_rcnn_ros::MaskedObjectAttributes> moas_msg_list;
	visualization_msgs::MarkerArray markers_list;
};

#endif
