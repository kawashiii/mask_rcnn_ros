#include <iostream>
#include <string>
#include <vector>

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

//PCL
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing_rgb.h>

//OpenCv
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//Msg or Srv
#include "mask_rcnn_ros/GetMaskedSurface.h"

using namespace std;

sensor_msgs::CameraInfoConstPtr camera_info;
cv::Mat cameraMatrix;
cv::Mat distCoeffs;
cv::Mat depth;
double fx;
double fy;
double cx;
double cy;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

ros::Publisher vis_axes_marker;
std::string camera_info_topic = "/pylon_camera_node/camera_info";
std::string depth_topic = "/phoxi_camera/aligned_depth_map";
std::string frame_id = "basler_ace_rgb_sensor_calibrated";

struct CenterNormalMsg{
    geometry_msgs::PointStamped center;
    geometry_msgs::Vector3Stamped normal;
};

bool compare_indices_size(const pcl::PointIndices& lp, const pcl::PointIndices& rp)
{
    return lp.indices.size() > rp.indices.size();
}

visualization_msgs::Marker build_marker_msg(std_msgs::Header header, string ns, int id, int type, int action, geometry_msgs::Vector3 scale, std_msgs::ColorRGBA color, geometry_msgs::Point point, geometry_msgs::Vector3 vector, string text)
{
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = ns;
    marker.id = id;
    marker.type = type;
    marker.action = action;
    marker.scale = scale;
    marker.color = color;
    if (type == visualization_msgs::Marker::TEXT_VIEW_FACING)
    {
        marker.pose.position.x = point.x + vector.x * 0.05;
        marker.pose.position.y = point.y + vector.y * 0.05;
        marker.pose.position.z = point.z + vector.z * 0.05 + 0.01;
        marker.text = text;

        return marker;
    }
    marker.points.resize(2);
    marker.points[0] = point;
    marker.points[1].x = point.x + vector.x * 0.05;
    marker.points[1].y = point.y + vector.y * 0.05;
    marker.points[1].z = point.z + vector.z * 0.05;
    marker.text = text;

    return marker;
}

CenterNormalMsg getRegionGrowing(cv::Mat mask_index)
{
    CenterNormalMsg msg;

    PointCloudT::Ptr cloud (new PointCloudT);
    for (int i = 0; i < mask_index.total(); i++)
    {
        int u = mask_index.at<cv::Point>(i).x;
	int v = mask_index.at<cv::Point>(i).y;
	PointT p;
	p.z = depth.at<float>(v, u) / 1000;
	p.x = (u - cx) * p.z / fx;
	p.y = (v - cy) * p.z / fy;
	cloud->points.push_back(p);
    }

    //downsampling
    pcl::VoxelGrid<PointT> voxelSampler;
    voxelSampler.setInputCloud(cloud);
    voxelSampler.setLeafSize(0.002, 0.002, 0.002);
    voxelSampler.filter(*cloud);

    //regiongrowing
    pcl::search::Search<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    pcl::PointCloud <pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<PointT, pcl::Normal> normal_estimator;
    pcl::RegionGrowing<PointT, pcl::Normal> reg;
    vector<pcl::PointIndices> indices;
    normal_estimator.setSearchMethod (tree);
    normal_estimator.setInputCloud (cloud);
    normal_estimator.setKSearch (50);
    normal_estimator.compute (*cloud_normals);
    reg.setMinClusterSize (100);
    reg.setMaxClusterSize (10000);
    reg.setSearchMethod (tree);
    reg.setNumberOfNeighbours (30);
    reg.setInputCloud (cloud);
    reg.setInputNormals (cloud_normals);
    reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold (1.0);
    reg.extract(indices);

    if (indices.size() == 0) {
	ROS_WARN("Couldn't find any surface");
        return msg;
    }
    ROS_INFO("%d surfaces found", (int)indices.size());

    sort(indices.begin(), indices.end(), compare_indices_size);
    
    vector<int> max_size_indices = indices[0].indices;
    pcl::CentroidPoint<PointT> centroid;
    PointCloudT::Ptr tmp(new PointCloudT);
    PointT c1;
    for (auto point = max_size_indices.begin(); point != max_size_indices.end(); point++)
    {
	centroid.add(cloud->points[*point]);
    }
    centroid.get(c1);

    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);
    int K = 1;
    vector<int> indices_(K);
    vector<float> distance(K);
    kdtree.nearestKSearch(c1, K, indices_, distance);

    msg.center.header.frame_id = frame_id;
    msg.center.header.stamp = ros::Time::now();
    msg.center.point.x = cloud->points[indices_[0]].x;
    msg.center.point.y = cloud->points[indices_[0]].y;
    msg.center.point.z = cloud->points[indices_[0]].z;
    msg.normal.header.frame_id = frame_id;
    msg.normal.header.stamp = ros::Time::now();
    msg.normal.vector.x = cloud_normals->points[indices_[0]].normal_x;
    msg.normal.vector.y = cloud_normals->points[indices_[0]].normal_y;
    msg.normal.vector.z = cloud_normals->points[indices_[0]].normal_z;
    if (msg.normal.vector.z > 0) {
        msg.normal.vector.x *= -1;
        msg.normal.vector.y *= -1;
        msg.normal.vector.z *= -1;
    }

    return msg;
}

bool callbackGetLocalSurface(mask_rcnn_ros::GetMaskedSurface::Request &req, mask_rcnn_ros::GetMaskedSurface::Response &res)
{
    ROS_INFO("Service called");
    ros::WallTime start_process_time = ros::WallTime::now();
        
    visualization_msgs::MarkerArray marker_axes_list;
    geometry_msgs::Vector3 arrow_scale;
    arrow_scale.x = 0.01, arrow_scale.y = 0.01, arrow_scale.z = 0.01;
    std_msgs::ColorRGBA z_axis_color;
    z_axis_color.r = 0.0, z_axis_color.g = 0.0, z_axis_color.b = 1.0, z_axis_color.a = 1.0;
    visualization_msgs::Marker del_msg = build_marker_msg(std_msgs::Header(), "", 0, 0, visualization_msgs::Marker::DELETEALL, geometry_msgs::Vector3(), std_msgs::ColorRGBA(), geometry_msgs::Point(), geometry_msgs::Vector3(), "");
    marker_axes_list.markers.push_back(del_msg);

    int count = 0;
    vector<sensor_msgs::Image> mask_msgs = req.masks;
    for (sensor_msgs::Image mask_msg : mask_msgs)
    {
        cv_bridge::CvImagePtr cv_ptr;
	cv_ptr = cv_bridge::toCvCopy(mask_msg, "8UC1");
	cv::Mat mask = cv_ptr->image.clone();

	cv::Mat mask_index;
	cv::findNonZero(mask, mask_index);

	ROS_INFO("ID:%d", count);

	CenterNormalMsg msg = getRegionGrowing(mask_index);
	res.centers.push_back(msg.center);
	res.normals.push_back(msg.normal);

	visualization_msgs::Marker z_axis_marker = build_marker_msg(msg.center.header, "mask_region_growing_normal", count, visualization_msgs::Marker::ARROW, visualization_msgs::Marker::ADD, arrow_scale, z_axis_color, msg.center.point, msg.normal.vector, "normal_" + to_string(count));
	marker_axes_list.markers.push_back(z_axis_marker);

	count++;
    }

    vis_axes_marker.publish(marker_axes_list);
    
    ros::WallTime end_process_time = ros::WallTime::now();
    double execution_process_time = (end_process_time - start_process_time).toNSec() * 1e-9;
    ROS_INFO("Total Service time(s): %.3f", execution_process_time);

    return true;    
}

void callbackDepth(const sensor_msgs::ImageConstPtr& depth_msg)
{
    ROS_INFO("Subscribed Depth Image");
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(depth_msg, "32FC1");
    depth = cv_ptr->image.clone();
}

void setCameraParameters()
{
    cv::Mat K(3, 3, CV_64FC1, (void *)camera_info->K.data());
    cv::Mat D(1, 5, CV_64FC1, (void *)camera_info->D.data());

    cameraMatrix = K;
    distCoeffs = D;
    fx = K.at<double>(0, 0);
    fy = K.at<double>(1, 1);
    cx = K.at<double>(0, 2);
    cy = K.at<double>(1, 2);
}

int main(int argc, char *argv[])
{
    //Initialize ROS
    ros::init (argc, argv, "mask_region_growing");
    ros::NodeHandle nh;

    ROS_INFO("Waiting camera info from %s", camera_info_topic.c_str());
    camera_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>(camera_info_topic);
    ROS_INFO("Received Camera Info");
    setCameraParameters();

    ros::Subscriber subDepth = nh.subscribe<sensor_msgs::Image>(depth_topic, 1, callbackDepth);

    ros::ServiceServer get_local_surface = nh.advertiseService(ros::this_node::getName() + "/get_masked_surface", callbackGetLocalSurface);

    vis_axes_marker = nh.advertise<visualization_msgs::MarkerArray>(ros::this_node::getName() + "/axes", 0, true);

    ros::spin();
}
