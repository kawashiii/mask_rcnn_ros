#ifndef REGION_GROWING_SEGMENTATION_H
#define REGION_GROWING_SEGMENTATION_H

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <math.h>

//PCL
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
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

using namespace std;

class RegionGrowingSegmentation {
    public:
        RegionGrowingSegmentation();

	void setPointCloud(PointCloudT::Ptr cloud_in);
	PointCloudT::Ptr getPointCloud();

        void downSampling(float x_leaf, float y_leaf, float z_leaf);

	void outlierRemove(int K, float stddev_mul_thresh);

	void normalEstimationKSearch(int K);
	void normalEstimationRadiusSearch(float radius);

	void segmentation(int min_cluster_size, int max_cluster_size, int nn, float smoothness_threshold, float curvature_threshold);

        void computeMomentOfInertia();

	float getArea(PointCloudT::Ptr cloud_in);

	void createPointCloudFromDepthMap(cv::Mat depth, cv::Mat cameraMatrix, cv::Mat distCoeffs, float scale);


    private:
	PointCloudT::Ptr point_cloud;
	NormalCloudT::Ptr normal_cloud;
	vector<pcl::PointIndices> indices;

};

#endif
