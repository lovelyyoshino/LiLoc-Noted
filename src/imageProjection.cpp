#include "utility.h"
#include "liloc/cloud_info.h"
#include <livox_ros_driver/CustomMsg.h>

// 定义Velodyne点云格式结构体，包含坐标、强度、环号和时间信息
struct VelodynePointXYZIRT {
    PCL_ADD_POINT4D // 添加x, y, z坐标
    PCL_ADD_INTENSITY; // 添加强度
    uint16_t ring; // 环号
    float time; // 时间戳
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 确保内存对齐
} EIGEN_ALIGN16;

// 注册Velodyne点云类型
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

// 定义Ouster点云格式结构体
struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity; // 强度
    uint32_t t; // 时间戳
    uint16_t reflectivity; // 反射率
    uint8_t ring; // 环号
    uint16_t noise; // 噪声
    uint32_t range; // 距离
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

// 注册Ouster点云类型
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// 定义Robosense点云格式结构体
struct RobosensePointXYZIRT {
    PCL_ADD_POINT4D
    float intensity; // 强度
    uint16_t ring; // 环号
    double timestamp; // 时间戳
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

// 注册Robosense点云类型
POINT_CLOUD_REGISTER_POINT_STRUCT(RobosensePointXYZIRT, 
      (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
      (uint16_t, ring, ring)(double, timestamp, timestamp)
)

// 定义Mulran数据集的点云格式结构体
struct MulranPointXYZIRT {
    PCL_ADD_POINT4D
    float intensity; // 强度
    uint32_t t; // 时间戳
    int ring; // 环号
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 }EIGEN_ALIGN16;

// 注册Mulran点云类型
 POINT_CLOUD_REGISTER_POINT_STRUCT (MulranPointXYZIRT,
     (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
     (uint32_t, t, t) (int, ring, ring)
 )

// 使用Velodyne点格式作为通用表示
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000; // 队列长度

class ImageProjection : public ParamServer { // 图像投影类继承自参数服务器
private:
    std::mutex imuLock; // IMU锁
    std::mutex odoLock; // 里程计锁

    ros::Subscriber subLaserCloud; // 激光点云订阅者
    ros::Publisher pubLaserCloud; // 激光点云发布者
    
    ros::Publisher pubExtractedCloud; // 提取后的点云发布者
    ros::Publisher pubLaserCloudInfo; // 点云信息发布者

    ros::Subscriber subImu; // IMU订阅者
    std::deque<sensor_msgs::Imu> imuQueue; // IMU消息队列

    ros::Subscriber subOdom; // 里程计订阅者
    std::deque<nav_msgs::Odometry> odomQueue; // 里程计消息队列

    std::deque<sensor_msgs::PointCloud2> cloudQueue; // 点云消息队列
    std::deque<livox_ros_driver::CustomMsg> cloudQueueLivox; // Livox点云消息队列

    sensor_msgs::PointCloud2 currentCloudMsg; // 当前点云消息
    livox_ros_driver::CustomMsg currentCloudMsgLivox; // 当前Livox点云消息

    double *imuTime = new double[queueLength]; // IMU时间数组
    double *imuRotX = new double[queueLength]; // IMU X轴旋转数组
    double *imuRotY = new double[queueLength]; // IMU Y轴旋转数组
    double *imuRotZ = new double[queueLength]; // IMU Z轴旋转数组

    int imuPointerCur; // 当前IMU指针
    bool firstPointFlag; // 第一个点标志
    Eigen::Affine3f transStartInverse; // 起始变换的逆矩阵

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn; // 输入激光点云
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn; // 临时Ouster点云
    pcl::PointCloud<MulranPointXYZIRT>::Ptr tmpMulranCloudIn; // 临时Mulran点云
    pcl::PointCloud<PointType>::Ptr   fullCloud; // 完整点云

    int deskewFlag; // 去畸变标志

    bool odomDeskewFlag; // 里程计去畸变标志
    float odomIncreX; // 里程计增量X
    float odomIncreY; // 里程计增量Y
    float odomIncreZ; // 里程计增量Z

    liloc::cloud_info cloudInfo; // 点云信息
    double timeScanCur; // 当前扫描时间
    double timeScanEnd; // 扫描结束时间
    std_msgs::Header cloudHeader; // 点云头部信息

public:
    ImageProjection(): deskewFlag(0) // 构造函数，初始化去畸变标志
    {
        // 订阅IMU和里程计话题
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());

        // 根据传感器类型选择相应的点云订阅方式
        if (sensor == SensorType::LIVOX) {
            subLaserCloud = nh.subscribe<livox_ros_driver::CustomMsg>(pointCloudTopic, 5, &ImageProjection::cloudHandlerLivox, this, ros::TransportHints().tcpNoDelay());
        }
        else {
            subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        }

        // 发布提取后的点云和点云信息
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("liloc/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = nh.advertise<liloc::cloud_info> ("liloc/deskew/cloud_info", 1);

        allocateMemory(); // 分配内存
        resetParameters(); // 重置参数

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR); // 设置PCL日志级别
    }

    void allocateMemory() { // 分配内存
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>()); // 初始化输入点云
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>()); // 初始化临时Ouster点云
        tmpMulranCloudIn.reset(new pcl::PointCloud<MulranPointXYZIRT>()); // 初始化临时Mulran点云
        fullCloud.reset(new pcl::PointCloud<PointType>()); // 初始化完整点云

        resetParameters(); // 重置参数
    }

    void resetParameters() { // 重置参数
        laserCloudIn->clear(); // 清空输入点云
        fullCloud->clear(); // 清空完整点云

        imuPointerCur = 0; // 重置IMU指针
        firstPointFlag = true; // 重置第一个点标志
        odomDeskewFlag = false; // 重置里程计去畸变标志

        for (int i = 0; i < queueLength; ++i) // 初始化IMU相关数组
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){} // 析构函数

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg) { // IMU消息处理函数
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg); // 转换IMU消息

        if (correct) { // 如果需要校正
            thisImu.linear_acceleration.x = thisImu.linear_acceleration.x * kAccScale; // 校正加速度X
            thisImu.linear_acceleration.y = thisImu.linear_acceleration.y * kAccScale; // 校正加速度Y
            thisImu.linear_acceleration.z = thisImu.linear_acceleration.z * kAccScale; // 校正加速度Z
        }

        std::lock_guard<std::mutex> lock1(imuLock); // 加锁以保护IMU队列
        imuQueue.push_back(thisImu); // 将IMU消息加入队列
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg) { // 里程计消息处理函数
        std::lock_guard<std::mutex> lock2(odoLock); // 加锁以保护里程计队列
        odomQueue.push_back(*odometryMsg); // 将里程计消息加入队列
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg) { // 点云消息处理函数
        if (!cachePointCloud(laserCloudMsg)) // 缓存点云
            return;

        if (!deskewInfo()) // 去畸变信息
            return;

        projectPointCloud(); // 投影点云

        publishClouds(); // 发布点云

        resetParameters(); // 重置参数
    }

    void cloudHandlerLivox(const livox_ros_driver::CustomMsg::ConstPtr &laserCloudMsg) { // Livox点云消息处理函数
        if (!cachePointCloudLivox(laserCloudMsg)) // 缓存Livox点云
            return;

        if (!deskewInfo()) // 去畸变信息
            return;

        projectPointCloud(); // 投影点云

        publishClouds(); // 发布点云

        resetParameters(); // 重置参数
    }

    bool cachePointCloudLivox(const livox_ros_driver::CustomMsg::ConstPtr &laserCloudMsg) { // 缓存Livox点云
        static bool first_scan = true; // 首次扫描标志
        static double last_time = 0.0; // 上一次时间
        static double first_scan_time = 0.0; // 首次扫描时间

        cloudQueueLivox.push_back(*laserCloudMsg); // 将Livox点云消息加入队列
        if (cloudQueueLivox.size() <= 2){ // 至少需要两个点云消息
            return false;
        }

        currentCloudMsgLivox = std::move(cloudQueueLivox.front()); // 获取当前点云消息
        cloudQueueLivox.pop_front(); // 移除已处理的消息

        double cur_time = currentCloudMsgLivox.header.stamp.toSec(); // 当前时间

        if (cur_time < last_time) { // 检查时间顺序
            ROS_WARN("Livox Cloud Loop .");
            cloudQueueLivox.clear(); // 清空队列
            last_time = cur_time; // 更新最后时间
        }

        if (first_scan) { // 如果是首次扫描
            first_scan_time = cur_time; // 记录首次扫描时间
            first_scan = false; // 更新状态
        }
            
        double time_offset = (cur_time - first_scan_time) * (double)(1000); // 计算时间偏移（毫秒）

        for (size_t i = 1; i < currentCloudMsgLivox.point_num; ++i) { // 遍历点云中的每个点
             if ((currentCloudMsgLivox.points[i].line < N_SCAN) && // 检查行号有效性
                ((currentCloudMsgLivox.points[i].tag & 0x30) == 0x10 || (currentCloudMsgLivox.points[i].tag & 0x30) == 0x00) &&
                !HasInf(currentCloudMsgLivox.points[i]) && !HasNan(currentCloudMsgLivox.points[i]) &&
                !IsNear(currentCloudMsgLivox.points[i], currentCloudMsgLivox.points[i - 1])) 
                {
                    PointXYZIRT point; // 创建点对象
                    point.x = currentCloudMsgLivox.points[i].x; // 设置x坐标
                    point.y = currentCloudMsgLivox.points[i].y; // 设置y坐标
                    point.z = currentCloudMsgLivox.points[i].z; // 设置z坐标

                    point.time = time_offset + currentCloudMsgLivox.points[i].offset_time * 1e-6;  // 毫秒

                    point.ring = currentCloudMsgLivox.points[i].line; // 设置环号

                    laserCloudIn->push_back(point); // 将点添加到输入点云中
                }
        }

        // 按照时间排序点云
        std::sort(laserCloudIn->points.begin(), laserCloudIn->points.end(), [](const PointXYZIRT& x, const PointXYZIRT& y) 
                        -> bool { return (x.time < y.time); });

        cloudHeader = currentCloudMsgLivox.header; // 更新点云头部信息
        timeScanCur = cloudHeader.stamp.toSec(); // 当前扫描时间
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time / (double)(1000); // 扫描结束时间

        first_scan = true; // 重置首次扫描标志
        last_time = cur_time; // 更新最后时间

        deskewFlag = 1; // 设置去畸变标志为1

        return true; // 返回成功
    }

    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg) { // 缓存普通点云
        cloudQueue.push_back(*laserCloudMsg); // 将点云消息加入队列
        if (cloudQueue.size() <= 2) // 至少需要两个点云消息
            return false;

        currentCloudMsg = std::move(cloudQueue.front()); // 获取当前点云消息
        cloudQueue.pop_front(); // 移除已处理的消息

        // 对于没有环号和时间通道的点云
        if (!have_ring_time_channel) {
            pcl::PointCloud<PointXYZIRT>::Ptr lidarCloudIn(new pcl::PointCloud<PointXYZIRT>());
            pcl::moveFromROSMsg(currentCloudMsg, *lidarCloudIn); // 从ROS消息转换为PCL点云

            bool halfPassed = false; // 半圈通过标志
            int cloudNum = lidarCloudIn->points.size(); // 点云数量

            double startOrientation = -atan2(lidarCloudIn->points[0].y, lidarCloudIn->points[0].x); // 开始方向
            double endOrientation = -atan2(lidarCloudIn->points[lidarCloudIn->points.size() - 1].y, lidarCloudIn->points[lidarCloudIn->points.size() - 1].x) + 2 * M_PI; // 结束方向
            if (endOrientation - startOrientation > 3 * M_PI) {
                endOrientation -= 2*M_PI; // 调整结束方向
            }
            else if (endOrientation - startOrientation < M_PI) {
                endOrientation += 2 * M_PI; // 调整结束方向
            }
            double orientationDiff = endOrientation - startOrientation; // 方向差值
            PointXYZIRT point; // 创建点对象
            for (int i = 0; i < cloudNum; ++i) { // 遍历点云中的每个点
                point.x = lidarCloudIn->points[i].x; // 设置x坐标
                point.y = lidarCloudIn->points[i].y; // 设置y坐标
                point.z = lidarCloudIn->points[i].z; // 设置z坐标
                float ori = -atan2(point.y, point.x); // 计算当前点的方向
                if (!halfPassed) { // 如果未经过半圈
                    if (ori < startOrientation - M_PI / 2) {
                        ori += 2 * M_PI; // 调整方向
                    } else if (ori > startOrientation + M_PI * 3 / 2) {
                        ori -= 2 * M_PI; // 调整方向
                    }
                    if (ori - startOrientation > M_PI) { // 判断是否经过半圈
                        halfPassed = true; // 更新状态
                    }
                } else {
                    ori += 2 * M_PI; // 调整方向
                    if (ori < endOrientation - M_PI * 3 / 2) {
                        ori += 2 * M_PI; // 调整方向
                    } else if (ori > endOrientation + M_PI / 2) {
                        ori -= 2 * M_PI; // 调整方向
                    }
                }
                float relTime = (ori - startOrientation) / orientationDiff; // 计算相对时间

                lidarCloudIn->points[i].time = 0.1 * relTime; // 设置时间
            }

            *laserCloudIn += *lidarCloudIn; // 合并点云

            deskewFlag = 1; // 设置去畸变标志为1
        }

        else { // 有环号和时间通道的点云
            if (sensor == SensorType::VELODYNE) {
                pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn); // 转换点云
            }
            else if (sensor == SensorType::OUSTER) {
                pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn); // 转换Ouster点云
                laserCloudIn->points.resize(tmpOusterCloudIn->size()); // 调整大小
                laserCloudIn->is_dense = tmpOusterCloudIn->is_dense; // 设置密度
                for (size_t i = 0; i < tmpOusterCloudIn->size(); i++) { // 遍历Ouster点云
                    auto &src = tmpOusterCloudIn->points[i];
                    auto &dst = laserCloudIn->points[i];
                    dst.x = src.x; // 设置x坐标
                    dst.y = src.y; // 设置y坐标
                    dst.z = src.z; // 设置z坐标
                    dst.intensity = src.intensity; // 设置强度
                    dst.ring = src.ring; // 设置环号
                    dst.time = src.t * 1e-9f; // 设置时间
                }
            }
            else if (sensor == SensorType::MULRAN) {
                pcl::moveFromROSMsg(currentCloudMsg, *tmpMulranCloudIn); // 转换Mulran点云
                laserCloudIn->points.resize(tmpMulranCloudIn->size()); // 调整大小
                laserCloudIn->is_dense = tmpMulranCloudIn->is_dense; // 设置密度
                for (size_t i = 0; i < tmpMulranCloudIn->size(); i++) { // 遍历Mulran点云
                    auto &src = tmpMulranCloudIn->points[i];
                    auto &dst = laserCloudIn->points[i];
                    dst.x = src.x; // 设置x坐标
                    dst.y = src.y; // 设置y坐标
                    dst.z = src.z; // 设置z坐标
                    dst.intensity = src.intensity; // 设置强度
                    dst.ring = src.ring; // 设置环号
                    dst.time = static_cast<float>(src.t); // 设置时间
                }
            }
            else if (sensor == SensorType::ROBOSENSE) {
                pcl::PointCloud<RobosensePointXYZIRT>::Ptr tmpRobosenseCloudIn(new pcl::PointCloud<RobosensePointXYZIRT>()); // 创建临时Robosense点云
                pcl::moveFromROSMsg(currentCloudMsg, *tmpRobosenseCloudIn); // 转换点云
                laserCloudIn->points.resize(tmpRobosenseCloudIn->size()); // 调整大小
                laserCloudIn->is_dense = tmpRobosenseCloudIn->is_dense; // 设置密度

                double start_stamptime = tmpRobosenseCloudIn->points[0].timestamp; // 获取起始时间
                for (size_t i = 0; i < tmpRobosenseCloudIn->size(); i++) { // 遍历Robosense点云
                    auto &src = tmpRobosenseCloudIn->points[i];
                    auto &dst = laserCloudIn->points[i];
                    dst.x = src.x; // 设置x坐标
                    dst.y = src.y; // 设置y坐标
                    dst.z = src.z; // 设置z坐标
                    dst.intensity = src.intensity; // 设置强度
                    dst.ring = src.ring; // 设置环号
                    dst.time = src.timestamp - start_stamptime; // 设置时间
                }
            }
            else {
                ROS_ERROR_STREAM("Unknown Sensor Type: " << int(sensor)); // 错误处理
                ros::shutdown(); // 关闭节点
            }

            static int ringFlag = 0; // 环号标志
            if (ringFlag == 0) { // 检查环号字段
                ringFlag = -1; // 标记为未找到
                for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i) {
                    if (currentCloudMsg.fields[i].name == "ring") {
                        ringFlag = 1; // 找到环号字段
                        break;
                    }
                }
                if (ringFlag == -1) { // 未找到环号字段
                    ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                    ros::shutdown(); // 关闭节点
                }
            }

            if (deskewFlag == 0) { // 检查时间字段
                deskewFlag = -1; // 标记为未找到
                for (auto &field : currentCloudMsg.fields) {
                    if (field.name == "time" || field.name == "t") {
                        deskewFlag = 1; // 找到时间字段
                        break;
                    }
                }
                if (deskewFlag == -1) // 未找到时间字段
                    ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
            }
        }

        cloudHeader = currentCloudMsg.header; // 更新点云头部信息
        timeScanCur = cloudHeader.stamp.toSec(); // 当前扫描时间
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time; // 扫描结束时间

        if (laserCloudIn->is_dense == false) { // 检查点云是否稠密
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!"); // 错误处理
            ros::shutdown(); // 关闭节点
        }

        return true; // 返回成功
    }

    bool deskewInfo() { // 去畸变信息处理
        std::lock_guard<std::mutex> lock1(imuLock); // 加锁以保护IMU队列
        std::lock_guard<std::mutex> lock2(odoLock); // 加锁以保护里程计队列

        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd) { // 检查IMU数据是否可用
            ROS_DEBUG("Waiting for IMU data ...");
            return false; // 返回失败
        }

        imuDeskewInfo(); // 处理IMU去畸变信息

        odomDeskewInfo(); // 处理里程计去畸变信息

        return true; // 返回成功
    }

    void imuDeskewInfo() { // IMU去畸变信息处理
        cloudInfo.imuAvailable = false; // 默认IMU不可用

        while (!imuQueue.empty()) { // 移除过期的IMU数据
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty()) // 如果IMU队列为空
            return;

        imuPointerCur = 0; // 重置IMU指针

        for (int i = 0; i < (int)imuQueue.size(); ++i) { // 遍历IMU队列
            sensor_msgs::Imu thisImuMsg = imuQueue[i]; // 获取IMU消息
            double currentImuTime = thisImuMsg.header.stamp.toSec(); // 当前IMU时间

            if (imuType) { // 如果有IMU类型
                // 获取此扫描的滚转、俯仰和偏航估计
                if (currentImuTime <= timeScanCur)
                    imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);
            }

            if (currentImuTime > timeScanEnd + 0.01) // 超出扫描结束时间
                break;

            if (imuPointerCur == 0) { // 如果是第一个IMU数据
                imuRotX[0] = 0; // 初始化X旋转
                imuRotY[0] = 0; // 初始化Y旋转
                imuRotZ[0] = 0; // 初始化Z旋转
                imuTime[0] = currentImuTime; // 设置时间
                ++imuPointerCur; // 指针前进
                continue;
            }

            // 获取角速度
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // 积分旋转
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1]; // 计算时间差
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff; // 更新X旋转
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff; // 更新Y旋转
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff; // 更新Z旋转
            imuTime[imuPointerCur] = currentImuTime; // 更新时间
            ++imuPointerCur; // 指针前进
        }

        --imuPointerCur; // 指针后退一位

        if (imuPointerCur <= 0) // 如果没有有效的IMU数据
            return;

        cloudInfo.imuAvailable = true; // 设置IMU可用标志
    }

    void odomDeskewInfo() { // 里程计去畸变信息处理
        cloudInfo.odomAvailable = false; // 默认里程计不可用
        static float sync_diff_time = (imuRate >= 300) ? 0.01 : 0.20; // 同步差异时间
        while (!odomQueue.empty()) { // 移除过期的里程计数据
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - sync_diff_time)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty()) // 如果里程计队列为空
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur) // 如果最早的里程计时间大于当前扫描时间
            return;

        // 获取扫描开始时的里程计数据
        nav_msgs::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i) { // 遍历里程计队列
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur) // 查找在当前扫描时间之前的里程计数据
                continue;
            else
                break;
        }

        tf::Quaternion orientation; // 四元数用于表示姿态
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation); // 转换四元数

        double roll, pitch, yaw; // 滚转、俯仰和偏航
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw); // 获取姿态角

        // 保存初始猜测位置和姿态
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true; // 设置里程计可用标志

        odomDeskewFlag = false; // 重置里程计去畸变标志

        if (odomQueue.back().header.stamp.toSec() < timeScanEnd) // 如果最后的里程计时间小于扫描结束时间
            return;

        nav_msgs::Odometry endOdomMsg; // 结束时的里程计数据

        for (int i = 0; i < (int)odomQueue.size(); ++i) { // 遍历里程计队列
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd) // 查找在扫描结束时间之前的里程计数据
                continue;
            else
                break;
        }

        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0]))) // 检查协方差
            return;

        // 计算起始和结束的变换矩阵
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation); // 转换结束时的四元数
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw); // 获取结束时的姿态角
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        Eigen::Affine3f transBt = transBegin.inverse() * transEnd; // 计算变换差

        float rollIncre, pitchIncre, yawIncre; // 增量
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre); // 获取增量

        odomDeskewFlag = true; // 设置里程计去畸变标志为true
    }

    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur) { // 查找旋转
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0; // 初始化旋转

        int imuPointerFront = 0; // 前指针
        while (imuPointerFront < imuPointerCur) { // 查找对应的IMU数据
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0) { // 如果超出范围或指针为0
            *rotXCur = imuRotX[imuPointerFront]; // 直接获取当前旋转
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } 
        else { // 插值计算
            int imuPointerBack = imuPointerFront - 1; // 后指针
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]); // 前插值比例
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]); // 后插值比例
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack; // 计算当前旋转
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur) { // 查找位置
        *posXCur = 0; *posYCur = 0; *posZCur = 0; // 初始化位置

        // 如果传感器移动相对较慢，例如步行速度，则位置去畸变似乎效果不明显，因此下面代码被注释掉。

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur); // 计算比例

        // *posXCur = ratio * odomIncreX; // 计算当前位置
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    PointType deskewPoint(PointType *point, double relTime) { // 去畸变点
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false) // 如果去畸变功能禁用
            return *point; // 返回原点

        double pointTime = timeScanCur + relTime; // 计算点的绝对时间

        float rotXCur, rotYCur, rotZCur; // 当前旋转
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur); // 查找当前旋转

        float posXCur, posYCur, posZCur; // 当前位置信息
        findPosition(relTime, &posXCur, &posYCur, &posZCur); // 查找当前位置信息

        if (firstPointFlag == true) { // 如果是第一个点
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse(); // 计算起始变换的逆矩阵
            firstPointFlag = false; // 更新状态
        }

        // 将点转换到起始位置
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur); // 计算最终变换
        Eigen::Affine3f transBt = transStartInverse * transFinal; // 计算变换差

        PointType newPoint; // 新点
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3); // 计算新点的x坐标
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3); // 计算新点的y坐标
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3); // 计算新点的z坐标
        newPoint.intensity = point->intensity; // 保留强度信息

        return newPoint; // 返回新点
    }

    void projectPointCloud() { // 投影点云
        int cloudSize = laserCloudIn->points.size(); // 获取点云大小

        for (int i = 0; i < cloudSize; ++i) { // 遍历点云中的每个点
            PointType thisPoint; // 当前点
            thisPoint.x = laserCloudIn->points[i].x; // 设置x坐标
            thisPoint.y = laserCloudIn->points[i].y; // 设置y坐标
            thisPoint.z = laserCloudIn->points[i].z; // 设置z坐标
            thisPoint.intensity = laserCloudIn->points[i].intensity; // 设置强度

            float range = pointDistance(thisPoint); // 计算点到传感器的距离
            if (range < lidarMinRange || range > lidarMaxRange) // 检查距离范围
                continue;

            int rowIdn = laserCloudIn->points[i].ring; // 获取环号
            if (rowIdn < 0 || rowIdn >= N_SCAN) // 检查环号有效性
                continue;

            if (rowIdn % downsampleRate != 0) // 下采样检查
                continue;

            if (i % point_filter_num != 0) // 点过滤检查
                continue;

            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time); // 去畸变处理

            fullCloud->push_back(thisPoint); // 将处理后的点添加到完整点云中
        }
    }
    
    void publishClouds() { // 发布点云
        cloudInfo.header = cloudHeader; // 更新点云信息头
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, fullCloud, cloudHeader.stamp, lidarFrame); // 发布去畸变后的点云
        pubLaserCloudInfo.publish(cloudInfo); // 发布点云信息
    }
};

int main(int argc, char** argv) { // 主函数
    ros::init(argc, argv, "liloc"); // 初始化ROS节点

    common_lib_ = std::make_shared<CommonLib::common_lib>("LiLoc"); // 创建公共库实例

    ImageProjection IP; // 创建图像投影实例
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m"); // 输出启动信息

    ros::MultiThreadedSpinner spinner(3); // 创建多线程spinner
    spinner.spin(); // 启动spinner
    
    return 0; // 返回成功
}
