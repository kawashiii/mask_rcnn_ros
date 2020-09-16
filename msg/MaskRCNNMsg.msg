std_msgs/Header header

uint32 count
uint32[] ids
float32[] areas

# Surface pose and corner points
geometry_msgs/Vector3Stamped[] x_axes
geometry_msgs/Vector3Stamped[] y_axes
geometry_msgs/Vector3Stamped[] z_axes
geometry_msgs/PolygonStamped[] polygons

# Save Value as x_axes
geometry_msgs/Vector3Stamped[] axes

# Bounding boxes in pixels
sensor_msgs/RegionOfInterest[] boxes

# Integer class IDs for each bounding box
int32[] class_ids

# String class IDs for each bouding box
string[] class_names

# Float probability scores of the class_id
float32[] scores

# Instance masks as Image
sensor_msgs/Image[] masks

# Center Point of mask
geometry_msgs/PointStamped[] centers

# Normal vectors
geometry_msgs/Vector3Stamped[] normals
