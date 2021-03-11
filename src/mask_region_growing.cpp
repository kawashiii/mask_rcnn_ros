#include "mask_region_growing.h"

using namespace std;

MaskRegionGrowingNode::MaskRegionGrowingNode():
    timeout(5.0),
    debug_mode(0),
    is_service_called(false),
    is_subscribed_depth(false),
    frame_id("basler_ace_rgb_sensor_calibrated"),
    x_min(-0.298), x_max(0.298), y_min(-0.200), y_max(0.200), z_min(0.005), z_max(0.300)
{
    ROS_INFO("Initialized Node");

    if (nh.hasParam("/mask_rcnn/debug_mode"))
    {
        nh.getParam("/mask_rcnn/debug_mode", debug_mode);
        if (debug_mode) {
            ROS_WARN("'mask_region_growing node' is Debug Mode");
        }
    }

    std::string depth_topic = "/phoxi_camera/aligned_depth_map_rect";
    if (debug_mode)
        depth_topic = "/debug" + depth_topic;
    depth_sub = nh.subscribe<sensor_msgs::Image>(depth_topic, 1, &MaskRegionGrowingNode::callbackDepth, this);

    get_masked_surface_srv = nh.advertiseService(ros::this_node::getName() + "/get_masked_surface", &MaskRegionGrowingNode::callbackGetMaskedSurface, this);

    markers_pub = nh.advertise<visualization_msgs::MarkerArray>(ros::this_node::getName() + "/markers", 0, true);
    scene_surface_pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>(ros::this_node::getName() + "/scene_surface_pointcloud", 0, true);
    masked_surface_pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>(ros::this_node::getName() + "/masked_surface_pointcloud", 0, true);
    input_pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>(ros::this_node::getName() + "/input_pointcloud", 0, true);
    masked_depth_map_pub = nh.advertise<sensor_msgs::Image>(ros::this_node::getName() + "/masked_depth_map", 0, true);

    MaskRegionGrowingNode::markerInitialization();
    
    ROS_INFO("Waiting camera info ....");
    camera_info = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/pylon_camera_node/camera_info", ros::Duration(timeout));
    MaskRegionGrowingNode::setCameraParameters();
    ROS_INFO("Set camera paramters");

    listener.waitForTransform("container", frame_id, ros::Time(0), ros::Duration(10.0));
    listener.lookupTransform("container", frame_id, ros::Time(0), tf_camera_to_container);
    //Eigen::Affine3d eigen_tf;
    tf::transformTFToEigen(tf_camera_to_container, eigen_tf);
    //matrix_camera_to_container = eigen_tf.matrix().cast<float>();
    ROS_INFO("Got tf for camera to container");
}

void
MaskRegionGrowingNode::run()
{
    ros::Rate loop_rate(10);
    while (ros::ok())
    {
        if (is_service_called)
	{
	    publishPointCloud();
	    publishMarkerArray();
	    publishMaskedDepthMap();
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
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix, imageSize, CV_32FC1, map_x, map_y);
}

void
MaskRegionGrowingNode::markerInitialization()
{
    x_axis_color.r = 1.0, x_axis_color.g = 0.0, x_axis_color.b = 0.0, x_axis_color.a = 1.0;
    y_axis_color.r = 0.0, y_axis_color.g = 1.0, y_axis_color.b = 0.0, y_axis_color.a = 1.0;
    z_axis_color.r = 0.0, z_axis_color.g = 0.0, z_axis_color.b = 1.0, z_axis_color.a = 1.0;
    polygon_color.r = 1.0, polygon_color.g = 1.0, polygon_color.b = 0.0, polygon_color.a = 1.0;
    center_color.r = 1.0, center_color.g = 0.0, center_color.b = 0.0, center_color.a = 1.0;
    text_color.r = 1.0, text_color.g = 1.0, text_color.b = 1.0, text_color.a = 1.0;

    arrow_scale.x = 0.005, arrow_scale.y = 0.005, arrow_scale.z = 0.005;
    sphere_scale.x = 0.005, sphere_scale.y = 0.005, sphere_scale.z = 0.005;
    text_scale.x = 0.02, text_scale.y = 0.02, text_scale.z = 0.02;
}


void
MaskRegionGrowingNode::callbackDepth(const sensor_msgs::ImageConstPtr& depth_msg)
{
    ROS_INFO("Subscribed Depth Image");
    ros::WallTime start_process_time = ros::WallTime::now();
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(depth_msg, "32FC1");

    depth = cv_ptr->image.clone();   
    //cv::Mat distorted_depth = cv_ptr->image.clone();
    //cv::remap(distorted_depth, depth, map_x, map_y, CV_INTER_NN);

    scene_reg.createPointCloudFromDepthMap(depth, cameraMatrix, 1000.0);
    //pcl::copyPointCloud(*(scene_reg.getPointCloud()), *input_scene);
    input_scene = scene_reg.getPointCloud();
    //scene_reg.transformPointCloud(matrix_camera_to_container);
    scene_reg.passThroughFilter(-0.4, 0.4, -0.3, 0.3, 1.25, 1.8); 
    scene_reg.downSampling();
    scene_reg.outlierRemove();

    scene_reg.normalEstimationKSearch();
    scene_reg_indices = scene_reg.segmentation();
    scene_surface_list.push_back(scene_reg.getSegmentedColoredCloud());

    ros::WallTime end_process_time = ros::WallTime::now();
    double execution_process_time = (end_process_time - start_process_time).toNSec() * 1e-9;
    ROS_INFO("PreProcessing time(s): %.3f", execution_process_time);
    ROS_INFO("Finished Preprocessing");

    is_subscribed_depth = true;
}

bool
MaskRegionGrowingNode::callbackGetMaskedSurface(mask_rcnn_ros_msgs::GetMaskedSurface::Request &req, mask_rcnn_ros_msgs::GetMaskedSurface::Response &res)
{
    ROS_INFO("Service called");
    ros::WallTime start_process_time = ros::WallTime::now();

    center_list = {};
    scene_surface_list = {};
    masked_surface_list = {};
    moas_msg_list = {};
    markers_list = {};

    // first compute scene region growing segmentation
    // scene_reg.normalEstimationKSearch();
    // vector<pcl::PointIndices> scene_reg_indices = scene_reg.segmentation();
    // scene_surface_list.push_back(scene_reg.getSegmentedColoredCloud());

    int count = 0;
    mask_msgs = req.masks;
    bool is_rigid_object = req.is_rigid_object;
    for (sensor_msgs::Image mask_msg : mask_msgs)
    {
        cv_bridge::CvImagePtr cv_ptr;
	cv_ptr = cv_bridge::toCvCopy(mask_msg, "8UC1");
	cv::Mat mask = cv_ptr->image.clone();
        float masked_depth_std = req.masked_depth_std[count];

	ROS_INFO("ID:%d", count);
	if (is_rigid_object)
	    maskedRegionGrowing(mask, masked_depth_std);
	    //maskedRegionGrowing_pt(mask, masked_depth_std);
	//else
	    //computerCenter(mask);

	count++;
    }

    setSurfacePriority();
    res.moas_list = moas_msg_list;

    ros::WallTime end_process_time = ros::WallTime::now();
    double execution_process_time = (end_process_time - start_process_time).toNSec() * 1e-9;
    ROS_INFO("Total Service time(s): %.3f", execution_process_time);

    is_service_called = true;
    is_subscribed_depth = false;
    return true;
}

void
MaskRegionGrowingNode::maskedRegionGrowing_pt(cv::Mat mask, float masked_depth_std)
{
    RegionGrowingSegmentation mask_reg;
    mask_reg.createPointCloudFromMaskedDepthMap(depth, mask, cameraMatrix, 1000.0);
    mask_reg.downSampling();
    mask_reg.outlierRemove();
    PointCloudT::Ptr mask_point_cloud = mask_reg.getPointCloud();
    MomentOfInertia moi = mask_reg.getMomentOfInertia(mask_point_cloud);
    //float normal_angle = acos(moi.minor_vectors(2)/sqrt(pow(moi.minor_vectors(0),2) + pow(moi.minor_vectors(1),2) + pow(moi.minor_vectors(2),2)));
    //cout << normal_angle << endl;

    // Wait until finished preprocess
    ros::Rate loop_rate(10);
    int count = 0;
    while (!is_subscribed_depth) {
        count += 1;
	if (count > 50) throw std::exception();
        loop_rate.sleep();
    }

    //if (normal_angle > 5.0) 
    if (masked_depth_std > 10.0) 
    {
        PointCloudT::Ptr scene_point_cloud = scene_reg.getPointCloud();
	NormalCloudT::Ptr scene_normal_cloud = scene_reg.getNormalCloud();
        mask_rcnn_ros_msgs::MaskedObjectAttributes moas_msg;
	moas_msg.surface_count = 0;
        for (auto i = scene_reg_indices.begin(); i != scene_reg_indices.end(); ++i)
        {
	    PointCloudT::Ptr cloud = scene_reg.getPointCloud(*i);
	    PointT center = scene_reg.getCenter(cloud);
	    int center_index = scene_reg.getNeighborPointIndex(center);
            center = scene_point_cloud->points[center_index];
            if (!checkPointRegion(center)) {
                ROS_WARN("  This surface's center point is outside of container");
                continue;
            }   
            center_list.push_back(center);

            cv::Mat tmp_tvec_rvec = (cv::Mat_<float>(1, 3) << 0.0, 0.0, 0.0);
            cv::Mat tmp_distCoeffs = (cv::Mat_<float>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);
            cv::Mat camera_point = (cv::Mat_<float>(1, 3) << center.x, center.y, center.z);
            std::vector<cv::Point2f> projected_point;
            cv::projectPoints(camera_point, tmp_tvec_rvec, tmp_tvec_rvec, cameraMatrix, tmp_distCoeffs, projected_point);
            if (mask.at<unsigned char>((int)projected_point[0].y, (int)projected_point[0].x) == 0.0) {
                ROS_WARN("  This surface's center point is outside of mask");
                continue;
            }

	    float area = scene_reg.getArea(cloud);
	    moi = scene_reg.getMomentOfInertia(cloud);
	    //MomentOfInertia moi = scene_reg.getMomentOfInertia(cloud);
	    
            mask_rcnn_ros_msgs::MaskedObjectAttributes moas_tmp_msg = build_moa_msg(scene_point_cloud->points[center_index], scene_normal_cloud->points[center_index], area, moi);
	    moas_msg.centers.push_back(moas_tmp_msg.centers[0]);
	    moas_msg.normals.push_back(moas_tmp_msg.normals[0]);
	    moas_msg.areas.push_back(moas_tmp_msg.areas[0]);
	    moas_msg.long_sides.push_back(moas_tmp_msg.long_sides[0]);
	    moas_msg.short_sides.push_back(moas_tmp_msg.short_sides[0]);
	    moas_msg.corners.push_back(moas_tmp_msg.corners[0]);
	    moas_msg.x_axes.push_back(moas_tmp_msg.x_axes[0]);
	    moas_msg.y_axes.push_back(moas_tmp_msg.y_axes[0]);
	    moas_msg.z_axes.push_back(moas_tmp_msg.z_axes[0]);  
            moas_msg.surface_count += 1;
        }
	moas_msg_list.push_back(moas_msg);
    }
    else
    {
	PointT center = mask_reg.getCenter(mask_point_cloud);
	int center_index = mask_reg.getNeighborPointIndex(center);
        center_list.push_back(mask_point_cloud->points[center_index]);
	float area = mask_reg.getArea(mask_point_cloud);

        // approximate normal vector because of flat object
        NormalT normal;
        normal.normal_x = 0.0;
        normal.normal_y = 0.0;
        normal.normal_z = 1.0;
        mask_rcnn_ros_msgs::MaskedObjectAttributes moas_msg;
	moas_msg = build_moa_msg(mask_point_cloud->points[center_index], normal, area, moi);
        if (checkPointRegion(mask_point_cloud->points[center_index])) {
            moas_msg.surface_count = 1;
        }
        else {
            moas_msg = {};
	    moas_msg.surface_count = 0;
            ROS_WARN("  This point is outside of container");	    
	}
	moas_msg_list.push_back(moas_msg);
    }
}

void
MaskRegionGrowingNode::maskedRegionGrowing(cv::Mat mask, float masked_depth_std)
{
    vector<PointCloudT::Ptr> cloud_list;
    RegionGrowingSegmentation mask_reg;
    mask_reg.createPointCloudFromMaskedDepthMap(depth, mask, cameraMatrix, 1000.0);
    mask_reg.downSampling();
    mask_reg.outlierRemove();
    mask_reg.normalEstimationKSearch();
    
    if (masked_depth_std > 3.0)
    {
        vector<pcl::PointIndices> mask_reg_indices = mask_reg.segmentation();
        if (mask_reg_indices.size() == 0) {
            ROS_WARN("Couldn't find any surface");
            mask_rcnn_ros_msgs::MaskedObjectAttributes moas_msg;
            moas_msg.surface_count = 0;
            moas_msg_list.push_back(moas_msg);
	    return;
        }
        masked_surface_list.push_back(mask_reg.getSegmentedColoredCloud());
    
        ROS_INFO("  %d surfaces found", (int)mask_reg_indices.size());
        for (auto i = mask_reg_indices.begin(); i != mask_reg_indices.end(); i++)
        {
            PointCloudT::Ptr segmented_point_cloud = mask_reg.getPointCloud(*i);
            cloud_list.push_back(segmented_point_cloud);
            ROS_INFO("  %d points surface", (int)segmented_point_cloud->points.size());
        }
    }
    else
    {
        PointCloudT::Ptr pd = mask_reg.getPointCloud();
        //masked_surface_list.push_back(pd);
        cloud_list.push_back(pd);
    }

    ros::Rate loop_rate(10);
    int count = 0;
    while (!is_subscribed_depth) {
        count += 1;
	if (count > 50) throw std::exception();
        loop_rate.sleep();
    }

    if (cloud_list.size() > 1) {
        mask_rcnn_ros_msgs::MaskedObjectAttributes moas_msg;
	moas_msg.surface_count = 0;
	for (int i = 0; i < cloud_list.size(); i++)
	{
	    PointT center = scene_reg.getCenter(cloud_list[i]);
	    int center_index = scene_reg.getNeighborPointIndex(center);
	    pcl::PointIndices cluster = scene_reg.getSegmentFromPoint(center_index);
	    if (cluster.indices.size() == 0) {
	        ROS_WARN("  Couldn't any surface where the center point belongs to");
	        continue;
	    }
	    
            PointCloudT::Ptr scene_point_cloud = scene_reg.getPointCloud();
	    NormalCloudT::Ptr scene_normal_cloud = scene_reg.getNormalCloud();
	    PointCloudT::Ptr segmented_point_cloud = scene_reg.getPointCloud(cluster);
	    center = scene_reg.getCenter(segmented_point_cloud);
	    center_index = scene_reg.getNeighborPointIndex(center);
            center_list.push_back(scene_point_cloud->points[center_index]);
	    float area = scene_reg.getArea(segmented_point_cloud);
	    MomentOfInertia moi = scene_reg.getMomentOfInertia(segmented_point_cloud);

            cv::Mat tmp_tvec_rvec = (cv::Mat_<float>(1, 3) << 0.0, 0.0, 0.0);
            cv::Mat tmp_distCoeffs = (cv::Mat_<float>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);
            cv::Mat camera_point = (cv::Mat_<float>(1, 3) << center.x, center.y, center.z);
            std::vector<cv::Point2f> projected_point;
            cv::projectPoints(camera_point, tmp_tvec_rvec, tmp_tvec_rvec, cameraMatrix, tmp_distCoeffs, projected_point);
            if (mask.at<unsigned char>((int)projected_point[0].y, (int)projected_point[0].x) == 0.0) {
                ROS_WARN("  This surface's center point is outside of mask");
                continue;
            }

	    mask_rcnn_ros_msgs::MaskedObjectAttributes moas_tmp_msg = build_moa_msg(scene_point_cloud->points[center_index], scene_normal_cloud->points[center_index], area, moi);
            if (!checkPointRegion(scene_point_cloud->points[center_index])) {
                ROS_WARN("  This surface's center point is outside of container");
                continue;
            }   

	    moas_msg.centers.push_back(moas_tmp_msg.centers[0]);
	    moas_msg.normals.push_back(moas_tmp_msg.normals[0]);
	    moas_msg.areas.push_back(moas_tmp_msg.areas[0]);
	    moas_msg.long_sides.push_back(moas_tmp_msg.long_sides[0]);
	    moas_msg.short_sides.push_back(moas_tmp_msg.short_sides[0]);
	    moas_msg.corners.push_back(moas_tmp_msg.corners[0]);
	    moas_msg.x_axes.push_back(moas_tmp_msg.x_axes[0]);
	    moas_msg.y_axes.push_back(moas_tmp_msg.y_axes[0]);
	    moas_msg.z_axes.push_back(moas_tmp_msg.z_axes[0]);  
            moas_msg.surface_count += 1;
	}
	moas_msg_list.push_back(moas_msg);

    } else {
        PointCloudT::Ptr mask_point_cloud = mask_reg.getPointCloud();
	NormalCloudT::Ptr mask_normal_cloud = mask_reg.getNormalCloud();
	PointT center = mask_reg.getCenter(cloud_list[0]);
	int center_index = mask_reg.getNeighborPointIndex(center);
        center_list.push_back(mask_point_cloud->points[center_index]);
	float area = mask_reg.getArea(cloud_list[0]);
	MomentOfInertia moi = mask_reg.getMomentOfInertia(cloud_list[0]);
        
        NormalT normal;
        normal.normal_x = 0.0;
        normal.normal_y = 0.0;
        normal.normal_z = 1.0;
        mask_rcnn_ros_msgs::MaskedObjectAttributes moas_msg;
	//moas_msg = build_moa_msg(mask_point_cloud->points[center_index], mask_normal_cloud->points[center_index], area, moi);
	moas_msg = build_moa_msg(mask_point_cloud->points[center_index], normal, area, moi);
        if (checkPointRegion(mask_point_cloud->points[center_index])) {
            moas_msg.surface_count = 1;
        }
        else {
            moas_msg = {};
	    moas_msg.surface_count = 0;
            ROS_WARN("  This point is outside of container");	    
	}
	moas_msg_list.push_back(moas_msg);
    }

}

mask_rcnn_ros_msgs::MaskedObjectAttributes
MaskRegionGrowingNode::build_moa_msg(PointT center_point, NormalT normal_axis, float area, MomentOfInertia moi)
{
    geometry_msgs::PointStamped center;
    geometry_msgs::Vector3Stamped normal;
    center.header.frame_id = frame_id;
    center.header.stamp = ros::Time::now();
    center.point.x = center_point.x;
    center.point.y = center_point.y;
    center.point.z = center_point.z;
    normal.header.frame_id = frame_id;
    normal.header.stamp = ros::Time::now();
    normal.vector.x = normal_axis.normal_x;
    normal.vector.y = normal_axis.normal_y;
    normal.vector.z = normal_axis.normal_z;
    if (normal.vector.z > 0) {
        normal.vector.x *= -1;
        normal.vector.y *= -1;
        normal.vector.z *= -1;
    }

    float normal_angle = acos(moi.minor_vectors(2)/sqrt(pow(moi.minor_vectors(0),2) + pow(moi.minor_vectors(1),2) + pow(moi.minor_vectors(2),2))) * (180.0 / M_PI);
    if (moi.minor_vectors(2) < 0)
        normal_angle = acos(-1 * moi.minor_vectors(2)/sqrt(pow(moi.minor_vectors(0),2) + pow(moi.minor_vectors(1),2) + pow(moi.minor_vectors(2),2))) * (180.0 / M_PI);
    ROS_INFO("  Normal angle is %f degree.", normal_angle);

    geometry_msgs::Vector3Stamped x_axis;
    geometry_msgs::Vector3Stamped y_axis;
    geometry_msgs::Vector3Stamped z_axis;
    x_axis.header.frame_id = frame_id;
    x_axis.header.stamp = ros::Time::now();
    x_axis.vector.x = moi.major_vectors(0);
    x_axis.vector.y = moi.major_vectors(1);
    x_axis.vector.z = moi.major_vectors(2);
    y_axis.header.frame_id = frame_id;
    y_axis.header.stamp = ros::Time::now();
    y_axis.vector.x = moi.middle_vectors(0);
    y_axis.vector.y = moi.middle_vectors(1);
    y_axis.vector.z = moi.middle_vectors(2);
    z_axis.header.frame_id = frame_id;
    z_axis.header.stamp = ros::Time::now();
    z_axis.vector.x = moi.minor_vectors(0);
    z_axis.vector.y = moi.minor_vectors(1);
    z_axis.vector.z = moi.minor_vectors(2);

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

    float long_side = abs(moi.max_point_OBB.x - moi.min_point_OBB.x);
    float short_side = abs(moi.max_point_OBB.y - moi.min_point_OBB.y);
    ROS_INFO("  Long side is %f mm.", long_side*1000);
    ROS_INFO("  Short side is %f mm.", short_side*1000);
	
    mask_rcnn_ros_msgs::MaskedObjectAttributes moas_msg;
    moas_msg.centers.push_back(center);
    moas_msg.normals.push_back(normal);
    moas_msg.areas.push_back(area);
    moas_msg.long_sides.push_back(long_side);
    moas_msg.short_sides.push_back(short_side);
    moas_msg.corners.push_back(corner);
    moas_msg.x_axes.push_back(x_axis);
    moas_msg.y_axes.push_back(y_axis);
    moas_msg.z_axes.push_back(z_axis);

    return moas_msg;
}

void
MaskRegionGrowingNode::setSurfacePriority()
{
    std::vector<mask_rcnn_ros_msgs::MaskedObjectAttributes> moas_msg_list_tmp = moas_msg_list;
    moas_msg_list = {};

    for (int i = 0; i < moas_msg_list_tmp.size(); i++)
    {
        if (moas_msg_list_tmp[i].surface_count == 0 or moas_msg_list_tmp[i].surface_count == 1)
        {
            moas_msg_list.push_back(moas_msg_list_tmp[i]);
            continue;
        }

        std::vector<float> areas;
        std::vector<float> normal_angles;
        for (int j = 0; j < moas_msg_list_tmp[i].surface_count; j++) {
            float normal_x = moas_msg_list_tmp[i].z_axes[j].vector.x;
            float normal_y = moas_msg_list_tmp[i].z_axes[j].vector.y;
            float normal_z = moas_msg_list_tmp[i].z_axes[j].vector.z;
            float normal_angle = acos(normal_z / sqrt(pow(normal_x, 2) + pow(normal_y, 2) + pow(normal_z, 2))) * (180.0 / M_PI);
            if (normal_z < 0)
                normal_angle = acos(-1 * normal_z / sqrt(pow(normal_x, 2) + pow(normal_y, 2) + pow(normal_z, 2))) * (180.0 / M_PI);
            normal_angles.push_back(normal_angle); 
            areas.push_back(moas_msg_list_tmp[i].areas[j]);
        }

        int areaMaxIndex = std::max_element(areas.begin(), areas.end()) - areas.begin();
        int n = moas_msg_list_tmp[i].surface_count;
        std::vector<int> p(n);
        std::vector<float> areas_tmp(n);
        std::vector<float> short_sides_tmp(n);
        std::vector<float> long_sides_tmp(n);
        std::vector<geometry_msgs::PointStamped> centers_tmp(n);
        std::vector<geometry_msgs::Vector3Stamped> normals_tmp(n);
        std::vector<geometry_msgs::Vector3Stamped> x_axes_tmp(n);
        std::vector<geometry_msgs::Vector3Stamped> y_axes_tmp(n);
        std::vector<geometry_msgs::Vector3Stamped> z_axes_tmp(n);
        std::vector<geometry_msgs::PolygonStamped> corners_tmp(n);
        std::iota(p.begin(), p.end(), 0);
        if (normal_angles[areaMaxIndex] < 50.0) {
            sort(p.begin(), p.end(), [&](int a, int b) { return areas[a] > areas[b]; });
        } else {
            sort(p.begin(), p.end(), [&](int a, int b) { return normal_angles[a] < normal_angles[b]; });
        }
            
        for (int j = 0; j < n; j++) {
            areas_tmp[j] = moas_msg_list_tmp[i].areas[p[j]];
            short_sides_tmp[j] = moas_msg_list_tmp[i].short_sides[p[j]];
            long_sides_tmp[j] = moas_msg_list_tmp[i].long_sides[p[j]];
            centers_tmp[j] = moas_msg_list_tmp[i].centers[p[j]];
            normals_tmp[j] = moas_msg_list_tmp[i].normals[p[j]];
            x_axes_tmp[j] = moas_msg_list_tmp[i].x_axes[p[j]];
            y_axes_tmp[j] = moas_msg_list_tmp[i].y_axes[p[j]];
            z_axes_tmp[j] = moas_msg_list_tmp[i].z_axes[p[j]];
            corners_tmp[j] = moas_msg_list_tmp[i].corners[p[j]];
        }
        moas_msg_list_tmp[i].areas = areas_tmp;
        moas_msg_list_tmp[i].short_sides = short_sides_tmp;
        moas_msg_list_tmp[i].long_sides = long_sides_tmp;
        moas_msg_list_tmp[i].centers = centers_tmp;
        moas_msg_list_tmp[i].normals = normals_tmp;
        moas_msg_list_tmp[i].x_axes = x_axes_tmp;
        moas_msg_list_tmp[i].y_axes = y_axes_tmp;
        moas_msg_list_tmp[i].z_axes = z_axes_tmp;
        moas_msg_list_tmp[i].corners = corners_tmp;
        
        moas_msg_list.push_back(moas_msg_list_tmp[i]);
    }
        
}

bool
MaskRegionGrowingNode::checkPointRegion(PointT point)
{
    Eigen::Vector3d eigen_point;
    eigen_point << point.x, point.y, point.z;

    eigen_point = eigen_tf.rotation()*eigen_point + eigen_tf.translation();
    return (eigen_point[0] > x_min &&
	    eigen_point[0] < x_max &&
	    eigen_point[1] > y_min &&
	    eigen_point[1] < y_max &&
	    eigen_point[2] > z_min &&
	    eigen_point[2] < z_max);
}

void
MaskRegionGrowingNode::publishPointCloud()
{
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
    // publish input pointcloud
    sensor_msgs::PointCloud2 input_pointcloud_msg;
    pcl::toROSMsg(*input_scene, input_pointcloud_msg);
    input_pointcloud_msg.header.frame_id = frame_id;
    input_pointcloud_msg.header.stamp = ros::Time::now();
    input_pointcloud_pub.publish(input_pointcloud_msg);
}

void
MaskRegionGrowingNode::publishMarkerArray()
{
    visualization_msgs::Marker del_marker = buildDelMarker();
    markers_list.markers.push_back(del_marker);
    std_msgs::Header header;
    int count = 0;
    for (int i = 0; i < moas_msg_list.size(); i++)
    {
        if (moas_msg_list[i].surface_count == 0) continue;
        for (int j = 0; j < moas_msg_list[i].surface_count; j++)
	{
            geometry_msgs::PointStamped center = moas_msg_list[i].centers[j];
            geometry_msgs::Vector3Stamped x_axis = moas_msg_list[i].x_axes[j];
            geometry_msgs::Vector3Stamped y_axis = moas_msg_list[i].y_axes[j];
            geometry_msgs::Vector3Stamped z_axis = moas_msg_list[i].z_axes[j];
            geometry_msgs::PolygonStamped corner = moas_msg_list[i].corners[j];

	    header = center.header;

            visualization_msgs::Marker x_axis_marker = buildArrowMarker(center.header, "mask_region_growing_x_axis_" + to_string(i), j, center.point, x_axis.vector, x_axis_color);
            visualization_msgs::Marker y_axis_marker = buildArrowMarker(center.header, "mask_region_growing_y_axis_" + to_string(i), j, center.point, y_axis.vector, y_axis_color);
            visualization_msgs::Marker z_axis_marker = buildArrowMarker(center.header, "mask_region_growing_z_axis_" + to_string(i), j, center.point, z_axis.vector, z_axis_color);
	    visualization_msgs::Marker text_marker = buildTextMarker(center.header, "mask_region_growing_text_" + to_string(i), j, center.point, to_string(i) + "_" + to_string(j), text_color);

            markers_list.markers.push_back(x_axis_marker);
            markers_list.markers.push_back(y_axis_marker);
            markers_list.markers.push_back(z_axis_marker);
            markers_list.markers.push_back(text_marker);

	    vector<geometry_msgs::Point> polygons;
	    for (int k = 0; k < corner.polygon.points.size(); k++)
	    {
		geometry_msgs::Point c;
		c.x = corner.polygon.points[k].x;
		c.y = corner.polygon.points[k].y;
		c.z = corner.polygon.points[k].z;

		polygons.push_back(c);
	    }
            visualization_msgs::Marker polygon_marker = buildSphereListMarker(corner.header, "mask_region_growing_corner_" + to_string(i), j, polygons, polygon_color);
            markers_list.markers.push_back(polygon_marker);

	    geometry_msgs::Point point;
	    point.x = center_list[count].x;
	    point.y = center_list[count].y;
	    point.z = center_list[count].z;
            visualization_msgs::Marker center_marker = buildSphereMarker(header, "mask_region_growing_center_" + to_string(i), j, point, center_color);
	    markers_list.markers.push_back(center_marker);

	    count++;
	}
    }

    markers_pub.publish(markers_list);
}

void
MaskRegionGrowingNode::publishMaskedDepthMap()
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

    for (PointT center : center_list)
    {
        cv::Mat tmp_tvec_rvec = (cv::Mat_<float>(1, 3) << 0.0, 0.0, 0.0);
        cv::Mat tmp_distCoeffs = (cv::Mat_<float>(1, 5) << 0.0, 0.0, 0.0, 0.0, 0.0);
        cv::Mat camera_point = (cv::Mat_<float>(1, 3) << center.x, center.y, center.z);
        std::vector<cv::Point2f> projected_point;
        cv::projectPoints(camera_point, tmp_tvec_rvec, tmp_tvec_rvec, cameraMatrix, tmp_distCoeffs, projected_point);
        cv::Point2d pt((int)projected_point[0].x, (int)projected_point[0].y);
        circle(vis_depth, pt, 10, cv::Scalar(255,0,0), -1, 4);
    }

    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = frame_id;
    sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(header, "bgr8", vis_depth).toImageMsg();

    masked_depth_map_pub.publish(image_msg);
}

visualization_msgs::Marker
MaskRegionGrowingNode::buildDelMarker()
{
    visualization_msgs::Marker marker;
    marker.action = visualization_msgs::Marker::DELETEALL;

    return marker;
}

visualization_msgs::Marker
MaskRegionGrowingNode::buildSphereMarker(std_msgs::Header header, string ns, int id, geometry_msgs::Point point, std_msgs::ColorRGBA color)
{
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = ns;
    marker.id = id;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale = sphere_scale;
    marker.color = color;
    marker.pose.position.x = point.x;
    marker.pose.position.y = point.y;
    marker.pose.position.z = point.z;

    return marker;
}

visualization_msgs::Marker
MaskRegionGrowingNode::buildSphereListMarker(std_msgs::Header header, string ns, int id, vector<geometry_msgs::Point> points, std_msgs::ColorRGBA color)
{
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = ns;
    marker.id = id;
    marker.type = visualization_msgs::Marker::SPHERE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale = sphere_scale;
    marker.color = color;
    marker.points = points;

    return marker;
}

visualization_msgs::Marker
MaskRegionGrowingNode::buildArrowMarker(std_msgs::Header header, string ns, int id, geometry_msgs::Point point, geometry_msgs::Vector3 vector, std_msgs::ColorRGBA color)
{
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = ns;
    marker.id = id;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale = arrow_scale;
    marker.color = color;
    marker.points.resize(2);
    marker.points[0] = point;
    marker.points[1].x = point.x + vector.x * 0.02;
    marker.points[1].y = point.y + vector.y * 0.02;
    marker.points[1].z = point.z + vector.z * 0.02;

    return marker;
}

visualization_msgs::Marker
MaskRegionGrowingNode::buildTextMarker(std_msgs::Header header, string ns, int id, geometry_msgs::Point point, string text, std_msgs::ColorRGBA color)
{
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = ns;
    marker.id = id;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale = text_scale;
    marker.color = color;
    marker.pose.position.x = point.x + 0.01;
    marker.pose.position.y = point.y + 0.01;
    marker.pose.position.z = point.z - 0.03;
    marker.text = text;

    return marker;
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "mask_region_growing_node");

    MaskRegionGrowingNode node;
    node.run();

    return 0;
}
