#pragma once

#ifndef _UTILITY_H_
#define _UTILITY_H_

#define PCL_NO_PRECOMPILE 

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <common_lib.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <opencv2/opencv.hpp>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

#include "math_tools.h"

using gtsam::symbol_shorthand::X; // Pose3 (x, y, z, r, p, y)
using gtsam::symbol_shorthand::V; // Vel   (xdot, ydot, zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax, ay, az, gx, gy, gz)

typedef pcl::PointXYZI PointType;
typedef pcl::PointCloud<PointType> Cloud;
typedef Cloud::Ptr CloudPtr;

// 定义一个包含额外信息的点结构体，包含位置、强度、滚转角、俯仰角、偏航角和时间戳
struct PointXYZIRPYT {
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;     // 滚转角
    float pitch;    // 俯仰角
    float yaw;      // 偏航角
    double time;    // 时间戳
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

// 注册自定义点类型到PCL
POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose; // 使用自定义点类型作为位姿点
typedef pcl::PointCloud<PointTypePose> Trajectory; // 定义轨迹点云类型
typedef Trajectory::Ptr TrajectoryPtr;

std::shared_ptr<CommonLib::common_lib> common_lib_; // 公共库指针

// 传感器类型枚举
enum class SensorType { VELODYNE, OUSTER, LIVOX, ROBOSENSE, MULRAN };

// 模式类型枚举
enum class ModeType { LIO, RELO };

const static inline int kSessionStartIdxOffset = 1000000; // 会话开始索引偏移量

const double kAccScale = 9.80665; // 加速度标定常数

// 参数服务器类，用于管理ROS参数
class ParamServer {
public:
    ros::NodeHandle nh; // ROS节点句柄

    std::string pointCloudTopic; // 点云主题
    std::string imuTopic;         // IMU主题
    std::string odomTopic;        // 里程计主题

    ModeType mode;                // 工作模式

    int numberOfCores;            // CPU核心数量

    std::string lidarFrame;       // 激光雷达坐标系
    std::string baselinkFrame;    // 基准链接坐标系
    std::string odometryFrame;    // 里程计坐标系
    std::string mapFrame;         // 地图坐标系

    std::string savePCDDirectory; // 保存PCD文件目录
    std::string saveSessionDirectory; // 保存会话数据目录

    SensorType sensor;            // 传感器类型
    int N_SCAN;                   // 扫描线数量
    int Horizon_SCAN;             // 水平扫描分辨率

    bool have_ring_time_channel;   // 是否有环形时间通道

    int downsampleRate;           // 下采样率
    int point_filter_num;         // 点过滤数量

    float lidarMinRange;          // 激光雷达最小范围
    float lidarMaxRange;          // 激光雷达最大范围

    int imuType;                  // IMU类型
    float imuRate;                // IMU频率
    float imuAccNoise;            // IMU加速度噪声
    float imuGyrNoise;            // IMU陀螺仪噪声
    float imuAccBiasN;            // IMU加速度偏置噪声
    float imuGyrBiasN;            // IMU陀螺仪偏置噪声
    float imuGravity;             // IMU重力值
    float imuRPYWeight;           // IMU RPY权重

    bool correct;                 // 是否进行校正

    std::vector<double> extRotV;  // 外部旋转向量
    std::vector<double> extRPYV;  // 外部RPY向量
    std::vector<double> extTransV; // 外部平移向量

    Eigen::Matrix3d extRot;       // 外部旋转矩阵
    Eigen::Matrix3d extRPY;       // 外部RPY矩阵

    Eigen::Vector3d extTrans;     // 外部平移向量
    Eigen::Quaterniond extQRPY;   // 外部RPY四元数

    float z_tollerance;           // Z轴容差
    float rotation_tollerance;    // 旋转容差

    std::string regMethod;        // 配准方法
    float ndtResolution;          // NDT分辨率
    float ndtEpsilon;             // NDT收敛阈值

    float timeInterval;           // 时间间隔

    double mappingProcessInterval; // 映射处理间隔
    
    float surroundingKeyframeMapLeafSize; // 周围关键帧地图叶子大小
    float mappingSurfLeafSize;          // 映射表面叶子大小

    float surroundingkeyframeAddingDistThreshold; // 添加周围关键帧距离阈值
    float surroundingkeyframeAddingAngleThreshold; // 添加周围关键帧角度阈值
    float surroundingKeyframeDensity; // 周围关键帧密度
    float surroundingKeyframeSearchRadius; // 周围关键帧搜索半径

    float globalMapVisualizationSearchRadius; // 全局地图可视化搜索半径
    float globalMapVisualizationPoseDensity; // 全局地图可视化位姿密度
    float globalMapVisualizationLeafSize; // 全局地图可视化叶子大小

    ~ParamServer() { } // 析构函数

    // 构造函数，初始化参数
    ParamServer() {
        nh.param<std::string>("System/pointCloudTopic", pointCloudTopic, "points_raw");
        nh.param<std::string>("System/imuTopic", imuTopic, "imu_correct");
        nh.param<std::string>("System/odomTopic", odomTopic, "odometry/imu");

        std::string modeStr;
        nh.param<std::string>("System/mode", modeStr, "lio");
        if (modeStr == "lio") {
            mode = ModeType::LIO; // 设置为LIO模式
        }
        else if (modeStr == "relo") {
            mode = ModeType::RELO; // 设置为RELO模式
        }
        else {
            ROS_ERROR_STREAM("Invalid Mode Type (must be either 'lio' or 'relo'): " << modeStr);
            ros::shutdown(); // 无效模式，关闭ROS
        }

        nh.param<int>("System/numberOfCores", numberOfCores, 4); // 获取CPU核心数量

        nh.param<std::string>("System/lidarFrame", lidarFrame, "base_link");
        nh.param<std::string>("System/baselinkFrame", baselinkFrame, "base_link");
        nh.param<std::string>("System/odometryFrame", odometryFrame, "odom");
        nh.param<std::string>("System/mapFrame", mapFrame, "map");

        nh.param<std::string>("System/savePCDDirectory", savePCDDirectory, "/Downloads/LOAM/");
        nh.param<std::string>("System/saveSessionDirectory", saveSessionDirectory, "/Downloads/LOAM/");

        std::string sensorStr;
        nh.param<std::string>("Sensors/sensor", sensorStr, " ");
        if (sensorStr == "velodyne")
        {
            sensor = SensorType::VELODYNE; // 设置为Velodyne传感器
        }
        else if (sensorStr == "ouster")
        {
            sensor = SensorType::OUSTER; // 设置为Ouster传感器
        }
        else if (sensorStr == "livox")
        {
            sensor = SensorType::LIVOX; // 设置为Livox传感器
        } 
        else if  (sensorStr == "robosense") {
            sensor = SensorType::ROBOSENSE; // 设置为Robosense传感器
        }
        else if (sensorStr == "mulran")
        {
            sensor = SensorType::MULRAN; // 设置为Mulran传感器
        } 
        else {
            ROS_ERROR_STREAM("Invalid Sensor Type (must be either 'velodyne' or 'ouster' or 'livox' or 'robosense' or 'mulran'): " << sensorStr);
            ros::shutdown(); // 无效传感器类型，关闭ROS
        }

        nh.param<int>("Sensors/N_SCAN", N_SCAN, 16); // 获取扫描线数量
        nh.param<int>("Sensors/Horizon_SCAN", Horizon_SCAN, 1800); // 获取水平扫描分辨率

        nh.param<bool>("Sensors/have_ring_time_channel", have_ring_time_channel, true); // 检查是否有环形时间通道

        nh.param<int>("Sensors/downsampleRate", downsampleRate, 1); // 获取下采样率
        nh.param<int>("Sensors/point_filter_num", point_filter_num, 3); // 获取点过滤数量

        nh.param<float>("Sensors/lidarMinRange", lidarMinRange, 1.0); // 获取激光雷达最小范围
        nh.param<float>("Sensors/lidarMaxRange", lidarMaxRange, 1000.0); // 获取激光雷达最大范围

        nh.param<int>("Sensors/imuType", imuType, 0); // 获取IMU类型
        nh.param<float>("Sensors/imuRate", imuRate, 500.0); // 获取IMU频率
        nh.param<float>("Sensors/imuAccNoise", imuAccNoise, 0.01); // 获取IMU加速度噪声
        nh.param<float>("Sensors/imuGyrNoise", imuGyrNoise, 0.001); // 获取IMU陀螺仪噪声
        nh.param<float>("Sensors/imuAccBiasN", imuAccBiasN, 0.0002); // 获取IMU加速度偏置噪声
        nh.param<float>("Sensors/imuGyrBiasN", imuGyrBiasN, 0.00003); // 获取IMU陀螺仪偏置噪声
        nh.param<float>("Sensors/imuGravity", imuGravity, 9.80511); // 获取IMU重力值
        nh.param<float>("Sensors/imuRPYWeight", imuRPYWeight, 0.01); // 获取IMU RPY权重

        nh.param<bool>("Sensors/correct", correct, false); // 获取是否进行校正的设置

        nh.param<std::vector<double>>("Sensors/extrinsicRot", extRotV, std::vector<double>()); // 获取外部旋转向量
        nh.param<std::vector<double>>("Sensors/extrinsicRPY", extRPYV, std::vector<double>()); // 获取外部RPY向量
        nh.param<std::vector<double>>("Sensors/extrinsicTrans", extTransV, std::vector<double>()); // 获取外部平移向量

        // 将外部参数转换为Eigen格式
        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        extQRPY = Eigen::Quaterniond(extRPY).inverse(); // 计算外部RPY的逆四元数

        nh.param<float>("Mapping/z_tollerance", z_tollerance, FLT_MAX); // 获取Z轴容差
        nh.param<float>("Mapping/rotation_tollerance", rotation_tollerance, FLT_MAX); // 获取旋转容差

        nh.param<std::string>("Mapping/regMethod", regMethod, "DIRECT1"); // 获取配准方法
        nh.param<float>("Mapping/ndtResolution", ndtResolution, 1.0); // 获取NDT分辨率
        nh.param<float>("Mapping/ndtEpsilon", ndtEpsilon, 0.01); // 获取NDT收敛阈值

        nh.param<float>("Mapping/timeInterval", timeInterval, 0.2); // 获取时间间隔

        nh.param<double>("Mapping/mappingProcessInterval", mappingProcessInterval, 0.15); // 获取映射处理间隔

        nh.param<float>("Mapping/mappingSurfLeafSize", mappingSurfLeafSize, 0.2); // 获取映射表面叶子大小
        nh.param<float>("Mapping/surroundingKeyframeMapLeafSize", surroundingKeyframeMapLeafSize, 0.4); // 获取周围关键帧地图叶子大小

        nh.param<float>("Mapping/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0); // 获取添加周围关键帧距离阈值
        nh.param<float>("Mapping/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2); // 获取添加周围关键帧角度阈值
        nh.param<float>("Mapping/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0); // 获取周围关键帧密度
        nh.param<float>("Mapping/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0); // 获取周围关键帧搜索半径

        nh.param<float>("Mapping/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3); // 获取全局地图可视化搜索半径
        nh.param<float>("Mapping/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0); // 获取全局地图可视化位姿密度
        nh.param<float>("Mapping/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0); // 获取全局地图可视化叶子大小

        usleep(100); // 暂停100微秒
    }

    // IMU消息转换函数，将输入的IMU消息进行坐标变换并返回新的IMU消息
    sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in) {
        sensor_msgs::Imu imu_out = imu_in; // 初始化输出IMU消息
        // 旋转加速度
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc; // 应用外部旋转
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();
        // 旋转陀螺仪
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr; // 应用外部旋转
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();

        if (imuType) {
            // 旋转滚转、俯仰、偏航
            Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
            Eigen::Quaterniond q_final = q_from * extQRPY; // 应用外部RPY四元数
            imu_out.orientation.x = q_final.x();
            imu_out.orientation.y = q_final.y();
            imu_out.orientation.z = q_final.z();
            imu_out.orientation.w = q_final.w();

            // 检查四元数有效性
            if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
            {
                ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!"); // 四元数无效，提示用户
                ros::shutdown(); // 关闭ROS
            }
        }

        return imu_out; // 返回转换后的IMU消息
    }
};

// 发布点云的模板函数
template<typename T>
sensor_msgs::PointCloud2 publishCloud(const ros::Publisher& thisPub, const T& thisCloud, ros::Time thisStamp, std::string thisFrame) {
    sensor_msgs::PointCloud2 tempCloud; // 创建临时点云消息
    pcl::toROSMsg(*thisCloud, tempCloud); // 转换PCL点云为ROS消息
    tempCloud.header.stamp = thisStamp; // 设置时间戳
    tempCloud.header.frame_id = thisFrame; // 设置坐标框架

    if (thisPub.getNumSubscribers() != 0) // 如果有订阅者
        thisPub.publish(tempCloud); // 发布点云消息

    return tempCloud; // 返回发布的点云消息
}

// 获取ROS时间的模板函数
template<typename T>
double ROS_TIME(T msg) {
    return msg->header.stamp.toSec(); // 返回消息的时间戳（秒）
}

// 从IMU消息中提取角速度的模板函数
template<typename T>
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z) {
    *angular_x = thisImuMsg->angular_velocity.x; // 提取X轴角速度
    *angular_y = thisImuMsg->angular_velocity.y; // 提取Y轴角速度
    *angular_z = thisImuMsg->angular_velocity.z; // 提取Z轴角速度
}

// 从IMU消息中提取加速度的模板函数
template<typename T>
void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z) {
    *acc_x = thisImuMsg->linear_acceleration.x; // 提取X轴加速度
    *acc_y = thisImuMsg->linear_acceleration.y; // 提取Y轴加速度
    *acc_z = thisImuMsg->linear_acceleration.z; // 提取Z轴加速度
}

// 从IMU消息中提取RPY角的模板函数
template<typename T>
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw) {
    double imuRoll, imuPitch, imuYaw; // 声明变量存储RPY角
    tf::Quaternion orientation; // 创建四元数对象
    tf::quaternionMsgToTF(thisImuMsg->orientation, orientation); // 将IMU四元数消息转换为TF四元数
    tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw); // 从四元数获取RPY角

    *rosRoll = imuRoll; // 存储滚转角
    *rosPitch = imuPitch; // 存储俯仰角
    *rosYaw = imuYaw; // 存储偏航角
}

// 点云变换函数，根据给定的变换应用到输入点云上
pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn){
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>()); // 创建输出点云

    int cloudSize = cloudIn->size(); // 获取输入点云大小
    cloudOut->resize(cloudSize); // 调整输出点云大小

    // 根据给定的位姿生成变换矩阵
    Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    
    const int numberOfCores = 8; // 并行处理的核心数量
    #pragma omp parallel for num_threads(numberOfCores) // OpenMP并行循环
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i]; // 获取输入点云中的点
        // 应用变换矩阵
        cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
        cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
        cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
        cloudOut->points[i].intensity = pointFrom.intensity; // 保留强度信息
    }
    return cloudOut; // 返回变换后的点云
}

// 将PCL点转换为GTSAM的Pose3对象
gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                        gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
}

// 将GTSAM的Pose3对象转换为PCL点
PointTypePose gtsamPose3ToPclPoint(gtsam::Pose3 point) {
    PointTypePose pose; // 创建PCL点对象
    pose.x = point.translation().x(); // 设置X坐标
    pose.y = point.translation().y(); // 设置Y坐标
    pose.z = point.translation().z(); // 设置Z坐标

    pose.roll = point.rotation().roll(); // 设置滚转角
    pose.pitch = point.rotation().pitch(); // 设置俯仰角
    pose.yaw = point.rotation().yaw(); // 设置偏航角

    return pose; // 返回转换后的PCL点
}

// 将PCL点转换为Eigen的Affine3f对象
Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint) { 
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw); // 返回变换矩阵
}

// 生成全局节点索引
int ungenGlobalNodeIdx (const int& _session_idx, const int& _idx_in_graph) {
    return (_idx_in_graph - 1) / (_session_idx * kSessionStartIdxOffset); // 计算全局节点索引
} // ungenGlobalNodeIdx

// 生成全局节点索引
int genGlobalNodeIdx (const int& _session_idx, const int& _node_offset) {
    return (_session_idx * kSessionStartIdxOffset) + _node_offset + 1; // 计算全局节点索引
} // genGlobalNodeIdx

// 生成锚节点索引
int genAnchorNodeIdx (const int& _session_idx) {
    return (_session_idx * kSessionStartIdxOffset); // 计算锚节点索引
} // genAnchorNodeIdx

// 写入顶点信息到字符串列表
void writeVertex(const int _node_idx, const gtsam::Pose3& _initPose, std::vector<std::string>& vertices_str){
    gtsam::Point3 t = _initPose.translation(); // 获取位姿的平移部分
    gtsam::Rot3 R = _initPose.rotation(); // 获取位姿的旋转部分

    // 格式化当前顶点信息
    std::string curVertexInfo {
        "VERTEX_SE3:QUAT " + std::to_string(_node_idx) + " "
         + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " " 
        + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " 
        + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

    vertices_str.emplace_back(curVertexInfo); // 将当前顶点信息添加到字符串列表
}

// 写入边信息到字符串列表
void writeEdge(const std::pair<int, int> _node_idx_pair, const gtsam::Pose3& _relPose, std::vector<std::string>& edges_str){
    gtsam::Point3 t = _relPose.translation(); // 获取相对位姿的平移部分
    gtsam::Rot3 R = _relPose.rotation(); // 获取相对位姿的旋转部分

    // 格式化当前边信息
    std::string curEdgeInfo {
        "EDGE_SE3:QUAT " + std::to_string(_node_idx_pair.first) + " " + std::to_string(_node_idx_pair.second) + " "
        + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " " 
        + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " 
        + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

    edges_str.emplace_back(curEdgeInfo); // 将当前边信息添加到字符串列表  
}

// 检查点是否包含无穷大
template <typename T>
inline bool HasInf(const T& p) {
  return (std::isinf(p.x) || std::isinf(p.y) || std::isinf(p.z)); // 判断点的各个坐标是否为无穷大
}

// 检查点是否包含NaN
template <typename T>
inline bool HasNan(const T& p) {
  return (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)); // 判断点的各个坐标是否为NaN
}

// 判断两个点是否接近
template <typename T>
inline bool IsNear(const T& p1, const T& p2) {
  return ((abs(p1.x - p2.x) < 1e-7) || (abs(p1.y - p2.y) < 1e-7) ||
          (abs(p1.z - p2.z) < 1e-7)); // 判断两个点的坐标是否在一定范围内接近
}

// 计算点到原点的距离
template<typename T> 
float pointDistance(const T& p) { 
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z); // 计算点到原点的欧几里得距离
}

// 计算两个点之间的距离
template<typename T>
float pointDistance(const T& p1, const T& p2) {
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z)); // 计算两点之间的欧几里得距离
}

#endif
