#include "mask_region_growing_node.h"

MaskRegionGrowingNode::MaskRegionGrowingNode():
    is_service_called(true)
{
    ROS_INFO("Initialized Node");

    depth_sub = nh.subscribe<sensor_msgs::Image>("/phoxi_camera/aligned_depth_map", 1, MaskRegionGrowingNode::callbackDepth);

    get_masked_surface_srv = nh.advertiseService(ros::this_node::getName() + "/get_masked_surface", MaskRegionGrowingNode::callbackGetMaskedSurface);

    vis_axes_marker = nh.advertise<visualization_msgs::MarkerArray>(ros::this_node::getName() + "/axes", 0, true);

    scene_surface_pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>(ros::this_node::getName() + "/scene_surface_pointcloud", 0, true);
    
    masked_surface_pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>(ros::this_node::getName() + "/masked_surface_pointcloud", 0, true);
    
    masked_surface_pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>(ros::this_node::getName() + "/masked_surface_pointcloud", 0, true);
    
    input_pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>(ros::this_node::getName() + "/input_pointcloud", 0, true);
    
    masked_depth_map_pub = nh.advertise<sensor_msgs::Image>(ros::this_node::getName() + "/masked_depth_map", 0, true);

    ROS_INFO("Waiting camera info");
    camera_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/pylon_camera_node/camerea_info");

    MaskRegionGrowingNode::setCameraParameters();
}

void
MaskRegionGrowingNode::run()
{
    ros::Rate loop_rate(10);
    while (ros::ok())
    {
        if (is_service_called)
	{
	
	    is_service_called = false;
	}

	ros::spinOnce();
	loop_rate.sleep();
    }

    ros::spin();
}

void 
MaskRegionGrowingNode::setCameraParameters()
{
    cv::Mat K(3, 3, CV_64FC1, (void *)camera_info->K.data());
    cv::Mat D(1, 5, CV_64FC1, (void *)camera_info->D.data());

    cameraMatrix = K;
    distCoeffs = D;
    fx = K.at<double>(0, 0);
    fy = K.at<double>(1, 1);
    cx = K.at<double>(0, 2);
    cy = K.at<double>(1, 2);

    cv::Size imageSize(camera_info->width, camera_info->height);
    cv::initUndistortRectifyMap(cameraMatrix, disCoeffs, cv::Mat(), cameraMatrix, imageSize, CV_32FC1, map_x, map_y;)
}


void
MaskRegionGrowingNode::callbackDepth(const sensor_msgs::ImageConstPtr& depth_msg)
{
    ROS_INFO("Subscribed Depth Image");
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv::bridge::toCvCopy(depth_msg, "32FC1")
    
    cv::Mat distorted_depth = cv_ptr->image.clone();
    cv::remap(distorted_depth, depth, map_x, map_y, CV_INTER_NN);

    scene_reg.createPointCloudFromDepthMap(depth, cameraMatrix, 1000.0);
    scene_reg.downSampling();
    scene_reg.outlierRemove();

    ROS_INFO("Finished Preprocessing");
}

bool
MaskRegionGrowingNode::callbackGetMaskedSurface(mask_rcnn_ros::GetMaskedSurface::Request &req, mask_rcnn_ros::GetMaskedSurface::Response &res)
{
    ROS_INFO("Service called");
    ros::WallTime start_process_time = ros::WallTime::now();

    // first compute scene region growing segmentation
    scene_reg.normalEstimationKSearch();
    scene_reg.segmentation();

    is_service_called = true;
    return true;
}

void
MaskRegionGrowingNode::markerInitialization()
{
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "mask_region_growing_node");

    MaskRegionGrowingNode node;
    node.run();

    return 0;
}
