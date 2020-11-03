#include "mask_region_growing_node.h"

MaskRegionGrowingNode::MaskRegionGrowingNode():
    is_service_called(true)
{
    ROS_INFO("Initialized Node");

    depth_sub = nh.subscribe<sensor_msgs::Image>("/phoxi_camera/aligned_depth_map", 1, MaskRegionGrowingNode::callbackDepth);

    get_masked_surface_srv = nh.advertiseService(ros::this_node::getName() + "/get_masked_surface", MaskRegionGrowingNode::callbackGetMaskedSurface);

    axes_marker_pub = nh.advertise<visualization_msgs::MarkerArray>(ros::this_node::getName() + "/axes", 0, true);
    scene_surface_pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>(ros::this_node::getName() + "/scene_surface_pointcloud", 0, true);
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

    scene_surface_list = {};
    masked_surface_list = {};
    moas_msg_list = {};
    marker_axes_list = {};

    // first compute scene region growing segmentation
    scene_reg.normalEstimationKSearch();
    scene_reg.segmentation();

    int count = 0;
    mask_msgs = req.masks;
    boos is_rigid_object = req.is_rigid_object;
    for (sensor_msgs::Image mask_msg : mask_msgs)
    {
        cv_bridge::CvImagePtr cv_ptr;
	cv_ptr = cv_bridge::toCvCopy(mask_msg, "8UC1");
	cv::Mat mask = cv_ptr->image.clone();

	ROS_INFO("ID:%d", count);
	if (is_rigid_object)
	    maskedRegionGrowing(mask);
	else
	    computerCenter(mask);

	count++;
    }

    is_service_called = true;
    return true;
}

void
MaskRegionGrowingNode::maskedRegionGrowing(cv::Mat mask)
{
    RegionGrowingSegmentation mask_reg;
    mask_reg.createPointCloudFromMaskedDepthMap(depth, mask, cameraMatrix);
    mask_reg.downSampling();
    mask_reg.outlierRemove();
    mask_reg.normalEstimationKSearch();
    pcl::PointIndices mask_reg_indices = mask_reg.segmentation();

    ROS_INFO("%d surfaces found", (int)mask_reg_indices.size());
    vector<PointCloudT::Ptr> cloud_list;
    vector<PointT> center_list;
    for (auto i = mask_reg_indices.begin(); i != mask_reg_indices.end(); i++)
    {
	PointCloudT::Ptr segmented_point_cloud = mask_reg.getPointCloud(i->indices);
	cloud_list.push_back(segmented_point_cloud);
    }

    if (cloud_list.size() > 1) {
	for (int i = 0; i < cloud_list.size(); i++)
	{
	    int center_index = mask_reg.getCenterIndex(cloud_list[i]);
	    pcl::PointIndices cluster = scene_reg.getSegmentFromPoint(center_index);

            PointCloudT::Ptr segmented_point_cloud = scene_reg.getPointCloud(cluster);
	    NormalCloudT::Ptr scene_normal_cloud = scene_reg.getNormalCloud();
	    center_index = scene_reg.getCenterIndex(segmented_point_cloud);
	    float area = scene_reg.getArea(segmented_point_cloud);
	    MomentOfInertia moi = scene_reg.getMomentOfInertia(segmented_point_cloud);

            mask_rcnn_ros::MaskedObjectAttributes moas_msg;
	    moas_msg = build_moa_msg(segmented_point_cloud, scene_normal_cloud, center_index, area, moi);
	    moas_msg_list.push_back(moas_msg);
	}
    } else {
        PointCloudT::Ptr mask_point_cloud = mask_reg.getPointCloud();
	NormalCloudT::Ptr mask_normal_cloud = mask_reg.getNormalCloud();
	int center_index = mask_reg.getCenterIndex(mask_point_cloud);
	float area = mask_reg.getArea(cloud_list[i]);
	MomentOfInertia moi = mask_reg.getMomentOfInertia(cloud_list[i]);
        
        mask_rcnn_ros::MaskedObjectAttributes moas_msg;
	moas_msg = build_moa_msg(mask_point_cloud, mask_normal_cloud, center_index, area, moi);
	
	moas_msg_list.push_back(moas_msg);
    }

}

mask_rcnn_ros::MaskedObjectAttributes
MaskRegionGrowingNode::build_moa_msg(PointCloudT::Ptr cloud, NormalCloudT::Ptr normal_cloud, int center_index, float area, MomentOfInertia moi)
{
    geometry_msgs::PointStamped center;
    geometry_msgs::Vector3Stamped normal;
    center.header.frame_id = frame_id;
    center.header.stamp = ros::Time::now();
    center.point.x = cloud->points[center_index].x;
    center.point.y = cloud->points[center_index].y;
    center.point.z = cloud->points[center_index].z;
    normal.header.frame_id = frame_id;
    normal.header.stamp = ros::Time::now();
    normal.vector.x = normal_cloud->points[center_index].normal_x;
    normal.vector.y = normal_cloud->points[center_index].normal_y;
    normal.vector.z = normal_cloud->points[center_index].normal_z;
    if (normal.vector.z > 0) {
        normal.vector.x *= -1;
        normal.vector.y *= -1;
        normal.vector.z *= -1;
    }

    vector<Eigen::Vector3f> axes = moi.vectors;
    geometry_msgs::Vector3Stamped x_axis;
    geometry_msgs::Vector3Stamped y_axis;
    geometry_msgs::Vector3Stamped z_axis;
    x_axis.header.frame_id = frame_id;
    x_axis.header.stamp = ros::Time::now();
    x_axis.vector.x = axes[0](0);
    x_axis.vector.y = axes[0](1);
    x_axis.vector.z = axes[0](2);
    y_axis.header.frame_id = frame_id;
    y_axis.header.stamp = ros::Time::now();
    y_axis.vector.x = axes[1](0);
    y_axis.vector.y = axes[1](1);
    y_axis.vector.z = axes[1](2);
    z_axis.header.frame_id = frame_id;
    z_axis.header.stamp = ros::Time::now();
    z_axis.vector.x = axes[2](0);
    z_axis.vector.y = axes[2](1);
    z_axis.vector.z = axes[2](2);
    if (z_axis.vector.z > 0) {
        z_axis.vector.x *= -1;
        z_axis.vector.y *= -1;
        z_axis.vector.z *= -1;
    }

    geometry_msgs::PolygonStamped corner;
    corner.header.frame_id = frame_id;
    corner.header.stamp = ros::Time::now();
    vector<geometry_msgs::Point32> corner_points;
    geometry_msgs::Point32 tmp;
    tmp.x = center.point.x + x_axis.vector.x * moi.min_point_OBB.x + y_axis.vector.x * moi.min_point_OBB.y;
    tmp.y = center.point.y + x_axis.vector.y * moi.min_point_OBB.x + y_axis.vector.y * moi.min_point_OBB.y;
    tmp.z = center.point.z + x_axis.vector.z * moi.min_point_OBB.x + y_axis.vector.z * moi.min_point_OBB.y;
    corner.polygon.points.push_back(tmp);
    tmp.x = center.point.x + x_axis.vector.x * moi.max_point_OBB.x + y_axis.vector.x * moi.min_point_OBB.y;
    tmp.y = center.point.y + x_axis.vector.y * moi.max_point_OBB.x + y_axis.vector.y * moi.min_point_OBB.y;
    tmp.z = center.point.z + x_axis.vector.z * moi.max_point_OBB.x + y_axis.vector.z * moi.min_point_OBB.y;
    corner.polygon.points.push_back(tmp);
    tmp.x = center.point.x + x_axis.vector.x * moi.min_point_OBB.x + y_axis.vector.x * moi.max_point_OBB.y;
    tmp.y = center.point.y + x_axis.vector.y * moi.min_point_OBB.x + y_axis.vector.y * moi.max_point_OBB.y;
    tmp.z = center.point.z + x_axis.vector.z * moi.min_point_OBB.x + y_axis.vector.z * moi.max_point_OBB.y;
    corner.polygon.points.push_back(tmp);
    tmp.x = center.point.x + x_axis.vector.x * moi.max_point_OBB.x + y_axis.vector.x * moi.max_point_OBB.y;
    tmp.y = center.point.y + x_axis.vector.y * moi.max_point_OBB.x + y_axis.vector.y * moi.max_point_OBB.y;
    tmp.z = center.point.z + x_axis.vector.z * moi.max_point_OBB.x + y_axis.vector.z * moi.max_point_OBB.y;
    corner.polygon.points.push_back(tmp);

    mask_rcnn_ros::MaskedObjectAttributes moas_msg;
    moas_msg.centers.push_back(center);
    moas_msg.normals.push_back(normal);
    moas_msg.areas.push_back(area);
    moas_msg.corners.push_back(corner);
    moas_msg.x_axes.push_back(x_axis);
    moas_msg.y_axes.push_back(y_axis);
    moas_msg.z_axes.push_back(z_axis);

    return moas_msg;
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
