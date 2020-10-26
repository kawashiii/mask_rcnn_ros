#include "region_growing_segmentation.h"

RegionGrowingSegmentation::RegionGrowingSegmentation():
    point_cloud(new PointCloudT),
    normal_cloud(new NormalCloudT)
{
}

void
RegionGrowingSegmentation::setPointCloud(PointCloudT::Ptr cloud_in)
{
    point_cloud->clear();
    pcl::copyPointCloud(*cloud_in, *point_cloud);
}

PointCloudT::Ptr
RegionGrowingSegmentation::getPointCloud()
{
    return point_cloud;
}

void
RegionGrowingSegmentation::createPointCloudFromDepthMap(cv::Mat depth, cv::Mat cameraMatrix, cv::Mat distCoeffs, float scale = 1.0)
{
}

void
RegionGrowingSegmentation::downSampling(float x_leaf = 0.002, float y_leaf = 0.002, float z_leaf = 0.002)
{
    pcl::VoxelGrid<PointT> voxelSampler;
    voxelSampler.setInputCloud(point_cloud);
    voxelSampler.setLeafSize(x_leaf, y_leaf, z_leaf);
    voxelSampler.filter(*point_cloud);
}

void
RegionGrowingSegmentation::outlierRemove(int K = 50, float stddev_mul_thresh = 1.0)
{
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(point_cloud);
    sor.setMeanK(K);
    sor.setStddevMulThresh(stddev_mul_thresh);
    sor.filter(*point_cloud);
}

void
RegionGrowingSegmentation::normalEstimationKSearch(int K = 30)
{
    normal_cloud->clear();
    pcl::search::Search<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    pcl::NormalEstimation<PointT, NormalT> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(point_cloud);
    normal_estimator.setKSearch(K);
    normal_estimator.compute(*normal_cloud);
}

void
RegionGrowingSegmentation::normalEstimationRadiusSearch(float radius = 0.02)
{
    normal_cloud->clear();
    pcl::search::Search<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    pcl::NormalEstimation<PointT, NormalT> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(point_cloud);
    normal_estimator.setRadiusSearch(radius);
    normal_estimator.compute(*normal_cloud);
}

void
RegionGrowingSegmentation::segmentation(int min_cluster_size = 100, int max_cluster_size = 7000, int nn = 30, float smoothness_threshold = 3.0/180.0*M_PI, float curvature_threshold = 1.0)
{
    pcl::RegionGrowing<PointT, pcl::Normal> reg;
    pcl::search::Search<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    reg.setMinClusterSize(min_cluster_size);
    reg.setMaxClusterSize(max_cluster_size);
    reg.setSearchMethod (tree);
    reg.setNumberOfNeighbours(nn);
    reg.setInputCloud(point_cloud);
    reg.setInputNormals(normal_cloud);
    reg.setSmoothnessThreshold(smoothness_threshold);
    reg.setCurvatureThreshold(curvature_threshold);
    reg.extract(indices);
}

void
RegionGrowingSegmentation::computeMomentOfInertia()
{
    PointT min_point_AABB;
    PointT max_point_AABB;
    PointT min_point_OBB;
    PointT max_point_OBB;
    PointT position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    Eigen::Vector3f major_vector, middle_vector, minor_vector;

    pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
    feature_extractor.setInputCloud(point_cloud);
    feature_extractor.compute();
    feature_extractor.getAABB(min_point_AABB, max_point_AABB);
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
    feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);
}

float
RegionGrowingSegmentation::getArea(PointCloudT::Ptr cloud_in)
{
    PointCloudT::Ptr cloud_out(new PointCloudT);

    pcl::ConvexHull<PointT> chull;
    chull.setInputCloud(cloud_in);
    chull.setComputeAreaVolume(true);
    chull.reconstruct(*cloud_out);

    return chull.getTotalArea();
}

