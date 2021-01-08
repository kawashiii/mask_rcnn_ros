#include "region_growing_segmentation.h"

using namespace std;

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
    PointCloudT::Ptr cloud_out(new PointCloudT);
    pcl::copyPointCloud(*point_cloud, *cloud_out);
    return cloud_out;
}

PointCloudT::Ptr
RegionGrowingSegmentation::getPointCloud(pcl::PointIndices cluster)
{
    PointCloudT::Ptr cloud_out(new PointCloudT);
    for (auto point = cluster.indices.begin(); point != cluster.indices.end(); point++)
    {
	cloud_out->points.push_back(point_cloud->points[*point]);
    }
    return cloud_out;
}

NormalCloudT::Ptr
RegionGrowingSegmentation::getNormalCloud()
{
    return normal_cloud;
}

void
RegionGrowingSegmentation::createPointCloudFromDepthMap(cv::Mat depth, cv::Mat cameraMatrix, float scale = 1000.0)
{
    int width = depth.cols;
    int height = depth.rows;
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);
    point_cloud->clear();
    point_cloud->width = width;
    point_cloud->height = height;
    point_cloud->resize(width * height);
    for (int v = 0; v < height; v++)
    {
        float *src = depth.ptr<float>(v);
	for (int u = 0; u < width; u++)
	{
	    PointT p;
	    p.z = src[u] / scale;
	    p.x = (u - cx) * p.z / fx;
	    p.y = (v - cy) * p.z / fy;
	    point_cloud->at(u, v) = p;
	}
    }
}

void
RegionGrowingSegmentation::createPointCloudFromMaskedDepthMap(cv::Mat depth, cv::Mat mask, cv::Mat cameraMatrix, float scale = 1000.0)
{
    int width = depth.cols;
    int height = depth.rows;
    double fx = cameraMatrix.at<double>(0, 0);
    double fy = cameraMatrix.at<double>(1, 1);
    double cx = cameraMatrix.at<double>(0, 2);
    double cy = cameraMatrix.at<double>(1, 2);

    cv::Mat mask_index;
    cv::findNonZero(mask, mask_index);

    point_cloud->clear();
    for (int i = 0; i < mask_index.total(); i++)
    {
        int u = mask_index.at<cv::Point>(i).x;
        int v = mask_index.at<cv::Point>(i).y;
	PointT p;
	p.z = depth.at<float>(v, u) / scale;
	p.x = (u - cx) * p.z / fx;
	p.y = (v - cy) * p.z / fy;
	//point_cloud->at(u, v) = p;
	point_cloud->points.push_back(p);
    }
}

void
RegionGrowingSegmentation::transformPointCloud(Eigen::Matrix4f matrix)
{
    pcl::transformPointCloud(*point_cloud, *point_cloud, matrix);
}

void
RegionGrowingSegmentation::passThroughFilter(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max)
{
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud(point_cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(x_min, x_max);
    pass.filter(*point_cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(y_min, y_max);
    pass.filter(*point_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(z_min, z_max);
    pass.filter(*point_cloud);
}

void
RegionGrowingSegmentation::downSampling(float x_leaf, float y_leaf, float z_leaf)
{
    pcl::VoxelGrid<PointT> voxelSampler;
    voxelSampler.setInputCloud(point_cloud);
    voxelSampler.setLeafSize(x_leaf, y_leaf, z_leaf);
    voxelSampler.filter(*point_cloud);
}

void
RegionGrowingSegmentation::outlierRemove(int K, float stddev_mul_thresh)
{
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(point_cloud);
    sor.setMeanK(K);
    sor.setStddevMulThresh(stddev_mul_thresh);
    sor.filter(*point_cloud);
}

void
RegionGrowingSegmentation::normalEstimationKSearch(int K)
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
RegionGrowingSegmentation::normalEstimationRadiusSearch(float radius)
{
    normal_cloud->clear();
    pcl::search::Search<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    pcl::NormalEstimation<PointT, NormalT> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(point_cloud);
    normal_estimator.setRadiusSearch(radius);
    normal_estimator.compute(*normal_cloud);
}

vector<pcl::PointIndices>
RegionGrowingSegmentation::segmentation(int min_cluster_size, int max_cluster_size, int nn, float smoothness_threshold, float curvature_threshold)
{
    pcl::search::Search<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    vector<pcl::PointIndices> indices;
    reg.setMinClusterSize(min_cluster_size);
    reg.setMaxClusterSize(max_cluster_size);
    reg.setSearchMethod (tree);
    reg.setNumberOfNeighbours(nn);
    reg.setInputCloud(point_cloud);
    reg.setInputNormals(normal_cloud);
    reg.setSmoothnessThreshold(smoothness_threshold);
    reg.setCurvatureThreshold(curvature_threshold);
    reg.extract(indices);

    return indices;
}

pcl::PointIndices
RegionGrowingSegmentation::getSegmentFromPoint(int index)
{
    pcl::PointIndices indices;
    reg.getSegmentFromPoint(index, indices);

    return indices;
}

PointT
RegionGrowingSegmentation::getCenter(PointCloudT::Ptr cloud_in)
{
    PointT center;
    pcl::computeCentroid(*cloud_in, center);
    
    return center;    
}

int
RegionGrowingSegmentation::getNeighborPointIndex(PointT point_in)
{
    vector<int> center_indices(1);
    vector<float> distances(1);
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(point_cloud);
    kdtree.nearestKSearch(point_in, 1, center_indices, distances);

    return center_indices[0];
}

PointCloudColorT::Ptr
RegionGrowingSegmentation::getSegmentedColoredCloud()
{
    return reg.getColoredCloud();
}

MomentOfInertia
RegionGrowingSegmentation::getMomentOfInertia(PointCloudT::Ptr cloud_in)
{
    MomentOfInertia moi;

    pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
    feature_extractor.setInputCloud(cloud_in);
    feature_extractor.compute();
    feature_extractor.getAABB(moi.min_point_AABB, moi.max_point_AABB);
    feature_extractor.getOBB(moi.min_point_OBB, moi.max_point_OBB, moi.position_OBB, moi.rotational_matrix_OBB);
    feature_extractor.getEigenVectors(moi.major_vectors, moi.middle_vectors, moi.minor_vectors);

    return moi;
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

