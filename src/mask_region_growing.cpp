#include <iostream>
#include <string>
#include <vector>
#include <cmath>

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
#include "mask_rcnn_ros/Centers.h"
#include "mask_rcnn_ros/Normals.h"

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979
#endif
#define deg_to_rad(deg) (((deg)/360)*2*M_PI)
#define rad_to_deg(rad) (((rad)/2/M_PI)*360)
sensor_msgs::CameraInfoConstPtr camera_info;
cv::Mat cameraMatrix;
cv::Mat distCoeffs;
cv::Mat depth;
vector<sensor_msgs::Image> mask_msgs;
double fx;
double fy;
double cx;
double cy;

typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB PointColorT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<PointColorT> PointCloudColorT;
PointCloudT::Ptr scene(new PointCloudT);
pcl::RegionGrowing<PointT, pcl::Normal> scene_reg;
pcl::PointCloud <pcl::Normal>::Ptr scene_normals (new pcl::PointCloud <pcl::Normal>);
vector<PointCloudColorT::Ptr> scene_surface_list;
vector<PointCloudColorT::Ptr> masked_surface_list;

ros::Publisher vis_axes_marker;
ros::Publisher masked_surface_pointcloud_pub;
ros::Publisher masked_surface_pointcloud_pub;
ros::Publisher drawed_depth_map_pub;
std::string camera_info_topic = "/pylon_camera_node/camera_info";
std::string depth_topic = "/phoxi_camera/aligned_depth_map";
std::string frame_id = "basler_ace_rgb_sensor_calibrated";
vector<mask_rcnn_ros::Centers> center_msg_list;
vector<mask_rcnn_ros::Normals> normal_msg_list;
//vector<geometry_msgs::PointStamped> center_msg_list;
//vector<geometry_msgs::Vector3Stamped> normal_msg_list;
visualization_msgs::MarkerArray marker_axes_list;
geometry_msgs::Vector3 arrow_scale;
std_msgs::ColorRGBA z_axis_color;

float is_service_called = false;

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
        marker.pose.position.x = point.x + vector.x * 0.04;
        marker.pose.position.y = point.y + vector.y * 0.04;
        marker.pose.position.z = point.z + vector.z * 0.04 + 0.01;
        marker.text = text;

        return marker;
    }
    marker.points.resize(2);
    marker.points[0] = point;
    marker.points[1].x = point.x + vector.x * 0.04;
    marker.points[1].y = point.y + vector.y * 0.04;
    marker.points[1].z = point.z + vector.z * 0.04;
    marker.text = text;

    return marker;
}

void sceneRegionGrowingFromPoint(vector<PointT> center_list)
{
    mask_rcnn_ros::Centers centers_msg;
    mask_rcnn_ros::Normals normals_msg;
    for (PointT center : center_list)
    {
        pcl::KdTreeFLANN<PointT> kdtree;
        kdtree.setInputCloud(scene);
        int K = 1;
        vector<int> indices_(K);
        vector<float> distance(K);
        kdtree.nearestKSearch(center, K, indices_, distance);

        pcl::PointIndices cluster;
        scene_reg.getSegmentFromPoint(indices_[0], cluster);

        pcl::CentroidPoint<PointT> centroid;
        PointT c1;
        for (auto point = cluster.indices.begin(); point != cluster.indices.end(); point++)
        {
            centroid.add(scene->points[*point]);
        }
        centroid.get(c1);
        if (c1.z == 0.0)
            continue;

        kdtree.setInputCloud(scene);
        kdtree.nearestKSearch(c1, K, indices_, distance);

        CenterNormalMsg msg;
        msg.center.header.frame_id = frame_id;
        msg.center.header.stamp = ros::Time::now();
        msg.center.point.x = scene->points[indices_[0]].x;
        msg.center.point.y = scene->points[indices_[0]].y;
        msg.center.point.z = scene->points[indices_[0]].z;
        msg.normal.header.frame_id = frame_id;
        msg.normal.header.stamp = ros::Time::now();
        msg.normal.vector.x = scene_normals->points[indices_[0]].normal_x;
        msg.normal.vector.y = scene_normals->points[indices_[0]].normal_y;
        msg.normal.vector.z = scene_normals->points[indices_[0]].normal_z;
        if (msg.normal.vector.z > 0) {
            msg.normal.vector.x *= -1;
            msg.normal.vector.y *= -1;
            msg.normal.vector.z *= -1;
        }

        centers_msg.centers.push_back(msg.center);
        normals_msg.normals.push_back(msg.normal);
    }
    center_msg_list.push_back(centers_msg);
    normal_msg_list.push_back(normals_msg);

    //if (centers_msg.centers.size() == 0) {
    //    ROS_WARN("Error region growing");
    //    return;
    //}
    //else if (centers_msg.centers.size() == 1)
    //{    
    //    center_msg_list.push_back(centers_msg);
    //    normal_msg_list.push_back(normals_msg);
    //}
    //else 
    //{
    //    for (int i = 0; i < centers_msg.centers.size(); i++) {
    //        geometry_msgs::PointStamped center = centers_msg.centers[i];
    //        geometry_msgs::Vector3Stamped normal = normals_msg.normals[i];
    //        // (a1b1 + a2b2 + a3b3)/sqrt(a1^2 + a2^2 + a3^2) * sqrt(b1^2 + b2^2 + b3^2)
    //        // a = (0, 0, 1), b = normal
    //        float cos_theta = -1 * normal.vector.z/sqrt(pow(normal.vector.x, 2.0) + pow(normal.vector.y, 2.0) + pow(normal.vector.z, 2.0));
    //        float theta = rad_to_deg(acos(cos_theta));
    //        if (theta < 30 && theta > -30) {
    //            centers_msg.centers = {};
    //            normals_msg.normals = {};
    //            centers_msg.centers.push_back(center);
    //            normals_msg.normals.push_back(normal);
    //            center_msg_list.push_back(centers_msg);
    //            normal_msg_list.push_back(normals_msg);
    //            return;
    //        }
    //    }
    //    ROS_WARN("Normal Vector is out of threshold");
    //}
}

void maskedRegionGrowing(cv::Mat mask_index)
{
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

    //outlier remover
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (30);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud);

    //regiongrowing
    pcl::search::Search<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    pcl::PointCloud <pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<PointT, pcl::Normal> normal_estimator;
    pcl::RegionGrowing<PointT, pcl::Normal> reg;
    vector<pcl::PointIndices> indices;
    normal_estimator.setSearchMethod (tree);
    normal_estimator.setInputCloud (cloud);
    normal_estimator.setKSearch (30);
    normal_estimator.compute (*cloud_normals);
    reg.setMinClusterSize (50);
    reg.setMaxClusterSize (7000);
    reg.setSearchMethod (tree);
    reg.setNumberOfNeighbours (30);
    reg.setInputCloud (cloud);
    reg.setInputNormals (cloud_normals);
    reg.setSmoothnessThreshold (10.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold (1.0);
    reg.extract(indices);

    if (indices.size() == 0) {
	ROS_WARN("Couldn't find any surface");
        return;
    }

    ROS_INFO("%d surfaces found", (int)indices.size());
    masked_surface_list.push_back(reg.getColoredCloud());
    vector<PointT> center_list;
    for (auto i = indices.begin(); i != indices.end(); i++)
    {
        pcl::CentroidPoint<PointT> centroid;
        PointT center;
        for (auto point = i->indices.begin(); point != i->indices.end(); point++)
        {
            centroid.add(cloud->points[*point]);
        }
        centroid.get(center);
        center_list.push_back(center);
    }

    mask_rcnn_ros::Centers centers_msg;
    mask_rcnn_ros::Normals normals_msg;
    if (center_list.size() > 1) {
        sceneRegionGrowingFromPoint(center_list);
    } else {
        PointT c1 = center_list[0];
        pcl::KdTreeFLANN<PointT> kdtree;
        kdtree.setInputCloud(cloud);
        int K = 1;
        vector<int> indices_(K);
        vector<float> distance(K);
        kdtree.nearestKSearch(c1, K, indices_, distance);

        CenterNormalMsg msg;
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

        //float cos_theta = -1 * msg.normal.vector.z/sqrt(pow(msg.normal.vector.x, 2.0) + pow(msg.normal.vector.y, 2.0) + pow(msg.normal.vector.z, 2.0));
        //float theta = rad_to_deg(acos(cos_theta));
        //std::cout << theta << std::endl;
        //if (theta > 25 || theta < -25) {
        //    ROS_WARN("Normal Vector is out of threshold");
        //    return;
        //}

        centers_msg.centers.push_back(msg.center);
        normals_msg.normals.push_back(msg.normal);

        center_msg_list.push_back(centers_msg);
        normal_msg_list.push_back(normals_msg);
    }
}

void markerInitialization()
{
    marker_axes_list = {};
    arrow_scale.x = 0.01, arrow_scale.y = 0.01, arrow_scale.z = 0.01;
    z_axis_color.r = 0.0, z_axis_color.g = 0.0, z_axis_color.b = 1.0, z_axis_color.a = 1.0;
    visualization_msgs::Marker del_msg = build_marker_msg(std_msgs::Header(), "", 0, 0, visualization_msgs::Marker::DELETEALL, geometry_msgs::Vector3(), std_msgs::ColorRGBA(), geometry_msgs::Point(), geometry_msgs::Vector3(), "");
    marker_axes_list.markers.push_back(del_msg);
}

void sceneRegionGrowing()
{
    pcl::search::Search<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    pcl::NormalEstimation<PointT, pcl::Normal> normal_estimator;
    vector<pcl::PointIndices> indices;
    normal_estimator.setSearchMethod (tree);
    normal_estimator.setInputCloud (scene);
    normal_estimator.setKSearch (30);
    normal_estimator.compute (*scene_normals);
    scene_reg.setMinClusterSize (50);
    scene_reg.setMaxClusterSize (3000);
    scene_reg.setSearchMethod (tree);
    scene_reg.setNumberOfNeighbours (30);
    scene_reg.setInputCloud (scene);
    scene_reg.setInputNormals (scene_normals);
    scene_reg.setSmoothnessThreshold (10.0 / 180.0 * M_PI);
    scene_reg.setCurvatureThreshold (1.0);
    scene_reg.extract(indices);
    scene_surface_list.push_back(scene_reg.getColoredCloud());
    ROS_INFO("Scene RegionGrowing finished");
}

void publishDrawedDepthMap()
{
    cv::Mat vis_depth;
    cv::Mat tmp;
    depth.convertTo(tmp, CV_8UC1);
    cv::cvtColor(tmp, vis_depth, CV_GRAY2BGR);
    for (sensor_msgs::Image mask_msg : mask_msgs)
    {
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(mask_msg, "8UC1");
        cv::Mat mask = cv_ptr->image.clone();

        cv::threshold(mask, mask, 125, 255, cv::THRESH_BINARY);
        vector<vector<cv::Point> > contours;
        vector<cv::Vec4i> hierarchy;
        cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        cv::drawContours(vis_depth, contours, -1, cv::Scalar(0, 0, 255), 4);

    }

    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = frame_id;
    sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(header, "bgr8", vis_depth).toImageMsg();
    
    drawed_depth_map_pub.publish(image_msg);
}

bool callbackGetMaskedSurface(mask_rcnn_ros::GetMaskedSurface::Request &req, mask_rcnn_ros::GetMaskedSurface::Response &res)
{
    ROS_INFO("Service called");
    ros::WallTime start_process_time = ros::WallTime::now();

    scene_surface_list = {};
    masked_surface_list = {};
    center_msg_list = {};
    normal_msg_list = {};
    markerInitialization();

    sceneRegionGrowing();
   
    int count = 0;
    mask_msgs = req.masks;
    for (sensor_msgs::Image mask_msg : mask_msgs)
    {
        cv_bridge::CvImagePtr cv_ptr;
	cv_ptr = cv_bridge::toCvCopy(mask_msg, "8UC1");
	cv::Mat mask = cv_ptr->image.clone();

        //offset
        //cv::Mat offset = (cv::Mat_<double>(2, 3) << 1.0, 0.0, 5.0, 0.0, 1.0, 0.0);
        //cv::warpAffine(mask, mask, offset, mask.size());

	cv::Mat mask_index;
	cv::findNonZero(mask, mask_index);

	ROS_INFO("ID:%d", count);
	maskedRegionGrowing(mask_index);

	count++;
    }

    res.centers_list = center_msg_list;
    res.normals_list = normal_msg_list;
    //for (int i = 0; i < center_msg_list.size(); i++)
    //{
    //    CenterNormalMsg msg; 
    //    msg.center = center_msg_list[i];
    //    msg.normal = normal_msg_list[i];
    //    visualization_msgs::Marker z_axis_marker = build_marker_msg(msg.center.header, "mask_region_growing_normal", i, visualization_msgs::Marker::ARROW, visualization_msgs::Marker::ADD, arrow_scale, z_axis_color, msg.center.point, msg.normal.vector, "normal_" + to_string(i));
    //    marker_axes_list.markers.push_back(z_axis_marker);
    //    res.centers.push_back(msg.center);
    //    res.normals.push_back(msg.normal);
    //}

    //vis_axes_marker.publish(marker_axes_list);
   
    ros::WallTime end_process_time = ros::WallTime::now();
    double execution_process_time = (end_process_time - start_process_time).toNSec() * 1e-9;
    ROS_INFO("Total Service time(s): %.3f", execution_process_time);

    is_service_called = true;

    return true;    
}

void callbackDepth(const sensor_msgs::ImageConstPtr& depth_msg)
{
    ROS_INFO("Subscribed Depth Image");
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(depth_msg, "32FC1");
    //depth = cv_ptr->image.clone();
    cv::Mat tmp = cv_ptr->image.clone();

    cv::Mat mapx, mapy;
    cv::Size imageSize(tmp.cols, tmp.rows);
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix, imageSize, CV_32FC1, mapx, mapy);
    cv::remap(tmp, depth, mapx, mapy, CV_INTER_NN);

    int width = depth.cols;
    int height = depth.rows;
    scene->width = width;
    scene->height = height;
    scene->resize(width*height);
    for (int v = 0; v < height; v++) {
        float *src = depth.ptr<float>(v);
        for (int u = 0; u < width; u++) {
            PointT p;
            p.z = src[u] / 1000;
            p.x = (u - cx) * p.z / fx;
            p.y = (v - cy) * p.z / fy;
            scene->at(u, v) = p;
        }
    }

    //downsampling
    pcl::VoxelGrid<PointT> voxelSampler;
    voxelSampler.setInputCloud(scene);
    voxelSampler.setLeafSize(0.002, 0.002, 0.002);
    voxelSampler.filter(*scene);

    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud (scene);
    sor.setMeanK (30);
    sor.setStddevMulThresh (1.0);
    sor.filter (*scene);

    //sensor_msgs::PointCloud2 output_cloud;
    //pcl::toROSMsg(*scene, output_cloud);
    //output_cloud.header.frame_id = frame_id;
    //output_cloud.header.stamp = ros::Time::now();

    //scene_surface_pointcloud_pub.publish(output_cloud);

    ROS_INFO("Preprocessed Finished");
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

    ros::ServiceServer get_local_surface = nh.advertiseService(ros::this_node::getName() + "/get_masked_surface", callbackGetMaskedSurface);

    vis_axes_marker = nh.advertise<visualization_msgs::MarkerArray>(ros::this_node::getName() + "/axes", 0, true);

    scene_surface_pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>(ros::this_node::getName() + "/scene_surface_pointcloud", 0, true);
    masked_surface_pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>(ros::this_node::getName() + "/masked_surface_pointcloud", 0, true);
    masked_depth_map_pub = nh.advertise<sensor_msgs::Image>(ros::this_node::getName() + "/masked_depth_map", 0, true);

    ros::Rate loop_rate(10);
    while(ros::ok())
    {
        if (is_service_called) {
            PointCloudColorT::Ptr scene_surfaces(new PointCloudColorT);
            PointCloudColorT::Ptr masked_surfaces(new PointCloudColorT);
            for (PointCloudColorT::Ptr scene_surface : scene_surface_list) {
                *scene_surfaces += *scene_surface;
            }
            for (PointCloudColorT::Ptr masked_surface : masked_surface_list) {
                *masked_surfaces += *masked_surface;
            }
            // publish scence surface
            sensor_msgs::PointCloud2 scene_surface_msg;
            pcl::toROSMsg(*scene_surfaces, scene_surface_msg);
            scene_surface_msg.header.frame_id = frame_id;
            scene_surface_msg.header.stamp = ros::Time::now();
            scene_surface_pointcloud_pub.publish(scene_surface_msg);
            // publish masked surface
            sensor_msgs::PointCloud2 masked_surface_msg;
            pcl::toROSMsg(*masked_surfaces, masked_surface_msg);
            masked_surface_msg.header.frame_id = frame_id;
            masked_surface_msg.header.stamp = ros::Time::now();
            masked_surface_pointcloud_pub.publish(masked_surface_msg);
            // publish aligned_depth_map drawed some information
            publishDrawedDepthMap();

            is_service_called = false;
            
        }
        ros::spinOnce();
        loop_rate.sleep();
    }

    ros::spin();
}
