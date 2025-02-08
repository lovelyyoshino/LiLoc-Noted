#ifndef _DATASAVER_HPP_  // 防止头文件重复包含
#define _DATASAVER_HPP_

#include <rosbag/bag.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_msgs/TFMessage.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TransformStamped.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/dataset.h>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/search/impl/search.hpp>

#include <fstream>
#include <iostream>

using namespace std;
using namespace gtsam;

using PointT = pcl::PointXYZI;  // 定义点云数据类型

namespace dataManager {

// 数据保存类定义，负责保存各种优化后的位姿、时间和点云数据等到指定目录下
class DataSaver {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // 对齐EIGEN数据结构的宏，用于避免内存对齐问题
    
    DataSaver() { }  // 默认构造函数

    ~DataSaver() { }  // 析构函数

    // 带参数构造函数，初始化数据保存目录
    DataSaver(string _base_dir, string _sequence_name) {
        this->base_dir = _base_dir;
        this->sequence_name = _sequence_name;

        if (_base_dir.back() != '/') {
            _base_dir.append("/");  // 确保路径以斜杠结尾
        }
        save_directory = _base_dir + sequence_name + '/';  // 拼接出完整的保存路径
        std::cout << "SAVE DIR:" << save_directory << std::endl;

        auto unused = system((std::string("exec rm -r ") + save_directory).c_str());  // 删除已有目录
        unused = system((std::string("mkdir -p ") + save_directory).c_str());  // 创建新的保存目录
    }

    // 设置保存目录
    void setDir(string _base_dir, string _sequence_name) {
        this->base_dir = _base_dir;
        this->sequence_name = _sequence_name;

        if (_base_dir.back() != '/') {
            _base_dir.append("/");
        }
        save_directory = _base_dir + sequence_name + '/';

        auto unused = system((std::string("exec rm -r ") + save_directory).c_str());
        unused = system((std::string("mkdir -p ") + save_directory).c_str());
    }

    // 设置配置目录
    void setConfigDir(string _config_dir) {
        if (_config_dir.back() != '/') {
            _config_dir.append("/");
        }
        this->config_directory = _config_dir;  // 更新配置目录
    }

    // 设置外参标定
    void setExtrinc(bool _use_imu, Eigen::Vector3d _t_body_sensor, Eigen::Quaterniond _q_body_sensor) {
        this->use_imu_frame = _use_imu;  // 是否使用IMU坐标系
        this->t_body_sensor = _t_body_sensor;  // 身体传感器的偏移量
        this->q_body_sensor = _q_body_sensor;  // 身体传感器的旋转四元数
    }

    // 保存优化后的KITTI格式位姿数据
    void saveOptimizedVerticesKITTI(gtsam::Values _estimates) {
        std::fstream stream(save_directory + "optimized_odom_kitti.txt", std::fstream::out);
        stream.precision(15);  // 设置浮点数精度
        for (const auto &key_value : _estimates) {
            auto p = dynamic_cast<const GenericValue<Pose3> *>(&key_value.value);
            if (!p) continue;  // 检查是否为Pose3类型

            const Pose3 &pose = p->value();  // 获取姿态

            Point3 t = pose.translation();  // 提取位置
            Rot3 R = pose.rotation();  // 提取旋转
            auto col1 = R.column(1);  // 获取旋转矩阵列
            auto col2 = R.column(2);
            auto col3 = R.column(3);

            // 将结果写入流中，包括旋转和位置信息
            stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x()
                   << " " << col1.y() << " " << col2.y() << " " << col3.y() << " "
                   << t.y() << " " << col1.z() << " " << col2.z() << " " << col3.z()
                   << " " << t.z() << std::endl;
        }
    }

    // 保存关键帧时间信息
    void saveTimes(vector<double> _keyframeTimes) {
        if (_keyframeTimes.empty()) {  // 如果时间向量为空则直接返回
            return;
        }
        this->keyframeTimes = _keyframeTimes;  // 存储时间节点
        std::fstream pgTimeSaveStream(save_directory + "times.txt",
                                  std::fstream::out);
        pgTimeSaveStream.precision(15);
        // 将每个时间戳写入文件
        for (auto const timestamp : keyframeTimes) {
            pgTimeSaveStream << timestamp << std::endl;
        }
        pgTimeSaveStream.close();
    }

    // 保存优化后的TUM格式位姿数据
    void saveOptimizedVerticesTUM(gtsam::Values _estimates) {
        std::fstream stream(save_directory + "optimized_odom_tum.txt", std::fstream::out);
        stream.precision(15);
        for (int i = 0; i < _estimates.size(); i++) {
            auto &pose = _estimates.at(i).cast<gtsam::Pose3>();
            gtsam::Point3 p = pose.translation();  // 提取位置
            gtsam::Quaternion q = pose.rotation().toQuaternion();  // 提取四元数
            // 将时间戳和位姿写入文件
            stream << keyframeTimes.at(i) << " " << p.x() << " " << p.y() << " "
                   << p.z() << " " << q.x() << " " << q.y() << " " << q.z() << " "
                   << q.w() << std::endl;
        }
    }

    // 保存GTSAM图优化结果
    void saveGraphGtsam(gtsam::NonlinearFactorGraph gtSAMgraph, gtsam::ISAM2 *isam, gtsam::Values isamCurrentEstimate) {
        gtsam::writeG2o(gtSAMgraph, isamCurrentEstimate, save_directory + "pose_graph.g2o");
        // 按照G2O格式将当前因子图和估计值写入文件
        gtsam::writeG2o(isam->getFactorsUnsafe(), isamCurrentEstimate, save_directory + "pose_graph.g2o");
    }

    // 保存里程计信息到g2o文件
    void saveGraph(std::vector<nav_msgs::Odometry> keyframePosesOdom) {
        std::fstream g2o_outfile(save_directory + "odom.g2o", std::fstream::out);
        g2o_outfile.precision(15);
        
        for (int i = 0; i < keyframePosesOdom.size(); i++) {
            nav_msgs::Odometry odometry = keyframePosesOdom.at(i);
            double time = odometry.header.stamp.toSec();  // 转换时间戳

            // 写入顶点位置及其变换
            g2o_outfile << "VERTEX_SE3:QUAT " << std::to_string(i) << " ";
            g2o_outfile << odometry.pose.pose.position.x << " ";
            g2o_outfile << odometry.pose.pose.position.y << " ";
            g2o_outfile << odometry.pose.pose.position.z << " ";
            g2o_outfile << odometry.pose.pose.orientation.x << " ";
            g2o_outfile << odometry.pose.pose.orientation.y << " ";
            g2o_outfile << odometry.pose.pose.orientation.z << " ";
            g2o_outfile << odometry.pose.pose.orientation.w << std::endl;
        }
        g2o_outfile.close();
    }

    // 保存激光测距和点云到ROS bag文件
    void saveResultBag(std::vector<nav_msgs::Odometry> allOdometryVec, std::vector<sensor_msgs::PointCloud2> allResVec) {
        rosbag::Bag result_bag;
        result_bag.open(save_directory + sequence_name + "_result.bag",
                    rosbag::bagmode::Write);

        // 写入里程计数据
        for (int i = 0; i < allOdometryVec.size(); i++) {
            nav_msgs::Odometry _laserOdometry = allOdometryVec.at(i);
            result_bag.write("pgo_odometry", _laserOdometry.header.stamp, _laserOdometry);
        }

        // 写入点云数据
        for (int i = 0; i < allResVec.size(); i++) {
            sensor_msgs::PointCloud2 _laserCloudFullRes = allResVec.at(i);
            result_bag.write("cloud_deskewed", _laserCloudFullRes.header.stamp, _laserCloudFullRes);
        }
        result_bag.close();
    }

    // 重载的保存方法，添加了变换数据保存
    void saveResultBag(std::vector<nav_msgs::Odometry> allOdometryVec, std::vector<sensor_msgs::PointCloud2> allResVec, std::vector<geometry_msgs::TransformStamped> trans_vec) {
        rosbag::Bag result_bag;
        result_bag.open(save_directory + sequence_name + "_result.bag", rosbag::bagmode::Write);

        tf2_msgs::TFMessage tf_message;  // 用于保存变换信息
        for (int i = 0; i < allOdometryVec.size(); i++) {
            nav_msgs::Odometry _laserOdometry = allOdometryVec.at(i);
            result_bag.write("pgo_odometry", _laserOdometry.header.stamp, _laserOdometry);

            sensor_msgs::PointCloud2 _laserCloudFullRes = allResVec.at(i);
            result_bag.write("cloud_deskewed", _laserCloudFullRes.header.stamp, _laserCloudFullRes);

            geometry_msgs::TransformStamped transform_stamped = trans_vec.at(i);
            tf_message.transforms.push_back(transform_stamped);  // 添加变换信息
            result_bag.write("tf", transform_stamped.header.stamp, tf_message);
        }
        result_bag.close();
    }

    // 保存整个点云地图到PCD文件
    void savePointCloudMap(std::vector<nav_msgs::Odometry> allOdometryVec, std::vector<pcl::PointCloud<PointT>::Ptr> allResVec) {
        std::cout << "odom and cloud size: " << allOdometryVec.size() << ", " << allResVec.size();

        int odom_size = std::min(allOdometryVec.size(), allResVec.size());  // 选择最小大小进行处理

        if (allOdometryVec.size() != allResVec.size()) {  // 比较尺寸确保匹配
            std::cout << " point cloud size do not equal to odom size!";
            return;
        }

        pcl::PointCloud<PointT>::Ptr laserCloudRaw(new pcl::PointCloud<PointT>());  // 原始点云
        pcl::PointCloud<PointT>::Ptr laserCloudTrans(new pcl::PointCloud<PointT>());  // 转换后的点云
        pcl::PointCloud<PointT>::Ptr globalmap(new pcl::PointCloud<PointT>());  // 全球地图
        for (int i = 0; i < odom_size; ++i) {
            nav_msgs::Odometry odom = allOdometryVec.at(i);  // 获取里程计信息
            laserCloudRaw = allResVec.at(i);  // 获取相应的点云

            Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();  // 初始化转换
            transform.rotate(Eigen::Quaterniond(odom.pose.pose.orientation.w, odom.pose.pose.orientation.x,
                                                odom.pose.pose.orientation.y, odom.pose.pose.orientation.z));
            transform.pretranslate(Eigen::Vector3d(odom.pose.pose.position.x,
                                                   odom.pose.pose.position.y,
                                                   odom.pose.pose.position.z));

            pcl::transformPointCloud(*laserCloudRaw, *laserCloudTrans, transform.matrix());  // 应用该变换
            *globalmap += *laserCloudTrans;  // 添加到全局地图
        }

        // 保存点云到 lidar 坐标系，如果你想在身体框架保存，可以启用此选项
        if (!globalmap->empty()) {
            globalmap->width = globalmap->points.size();  // 点云分辨率设置
            globalmap->height = 1;
            globalmap->is_dense = false;

            try {
                pcl::io::savePCDFileASCII(save_directory + "global_map_lidar.pcd", *globalmap);  // 保存PCD文件
                cout << "current scan saved to : " << save_directory << ", " << globalmap->points.size() << endl;
            } 
            catch (std::exception e) {
                ROS_ERROR_STREAM("SAVE PCD ERROR :" <<  globalmap->points.size());
            }

            // 所有云必须旋转至机身坐标轴
            if (use_imu_frame) {
                for (int j = 0; j < globalmap->points.size(); ++j) {
                    PointT &pt = globalmap->points.at(j);
                    Eigen::Vector3d translation(pt.x, pt.y, pt.z);
                    translation = q_body_sensor * translation + t_body_sensor;  // 应用传感器外参

                    pt.x = translation[0];
                    pt.y = translation[1];
                    pt.z = translation[2];
                }
                try {
                    pcl::io::savePCDFileASCII(save_directory + "globalmap_imu.pcd", *globalmap);  // 保存到 imu 坐标系的 PCD 文件
                    cout << "current scan saved to : " << save_directory << ", " << globalmap->points.size() << endl;
                } 
                catch (std::exception e) {
                    ROS_ERROR_STREAM("SAVE PCD ERROR :" <<  globalmap->points.size());
                }
            }
        } 
        else
            std::cout << "EMPTY POINT CLOUD";  // 如果没有点云输出空消息
    }

    // 另一个保存点云地图的方法
    void savePointCloudMap(pcl::PointCloud<PointT> &cloud_ptr) {
        if (cloud_ptr.empty()) {  // 如果点云为空输出提示
            std::cout << "empty global map cloud!" << std::endl;
            return;
        }
        try {
            pcl::io::savePCDFileASCII(save_directory + "globalmap_lidar_feature.pcd", cloud_ptr);  // 保存特征点云到 PCD 文件
        } 
        catch (pcl::IOException) {
            std::cout << "  save map failed!!! " << cloud_ptr.size() << std::endl;  // 错误捕获
        }
    }

private:
    string base_dir, sequence_name;  // 基础目录和序列名
    string save_directory, config_directory;  // 保存目录和配置目录

    vector<string> configParameter;  // 配置参数列表

    bool use_imu_frame = false;  // 使用IMU坐标系标志
    Eigen::Quaterniond q_body_sensor;  // 包含身体传感器的旋转的信息
    Eigen::Vector3d t_body_sensor;  // 身体传感器的平移偏移

    vector<double> keyframeTimes;  // 保留关键帧时间信息
};

}

#endif  // _DATASAVER_H_
