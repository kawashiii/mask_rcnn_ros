#ifndef REGION_GROWING_SEGMENTATION_H
#define REGION_GROWING_SEGMENTATION_H

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <math.h>

//PCL
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/surface/convex_hull.h>

//OpenCV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef pcl::PointXYZ PointT;
typedef pcl::Normal NormalT;
typedef pcl::PointXYZRGB PointColorT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<PointColorT> PointCloudColorT;
typedef pcl::PointCloud<NormalT> NormalCloudT;

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

struct MomentOfInertia
{
    PointT min_point_AABB;
    PointT max_point_AABB;
    PointT min_point_OBB;
    PointT max_point_OBB;
    PointT position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;

    Eigen::Vector3f major_vectors;
    Eigen::Vector3f middle_vectors;
    Eigen::Vector3f minor_vectors;
};

class RegionGrowingSegmentation {
    public:
        RegionGrowingSegmentation();

	void setPointCloud(PointCloudT::Ptr cloud_in);
	PointCloudT::Ptr getPointCloud();
	PointCloudT::Ptr getPointCloud(pcl::PointIndices indices);
	NormalCloudT::Ptr getNormalCloud();

	void transformPointCloud(Eigen::Matrix4f matrix);
	void passThroughFilter(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max);
        void downSampling(float x_leaf = 0.002, float y_leaf = 0.002, float z_leaf = 0.002);
	void outlierRemove(int K = 50, float stddev_mul_thresh = 1.0);
	void normalEstimationKSearch(int K = 30);
	void normalEstimationRadiusSearch(float radius = 0.02);
	std::vector<pcl::PointIndices> segmentation(int min_cluster_size = 100, int max_cluster_size = 50000, int nn = 30, float smoothness_threshold = 3.0/180*M_PI, float curvature_threshold = 1.0);

	pcl::PointIndices getSegmentFromPoint(int index);
	PointT getCenter(PointCloudT::Ptr cloud_in);
	int getNeighborPointIndex(PointT point_in);
	PointCloudColorT::Ptr getSegmentedColoredCloud();
	MomentOfInertia getMomentOfInertia(PointCloudT::Ptr cloud_in);
	float getArea(PointCloudT::Ptr cloud_in);

	void createPointCloudFromDepthMap(cv::Mat depth, cv::Mat cameraMatrix, float scale);
        void createPointCloudFromMaskedDepthMap(cv::Mat depth, cv::Mat mask, cv::Mat cameraMatrix, float scale);

    private:
	PointCloudT::Ptr point_cloud;
	NormalCloudT::Ptr normal_cloud;
	std::vector<pcl::PointIndices> indices;
        pcl::RegionGrowing<PointT, pcl::Normal> reg;
};

#endif
