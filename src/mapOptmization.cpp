#include "utility.h"
#include "tictoc.h"

#include "liloc/cloud_info.h"
#include "liloc/save_map.h"
#include "liloc/save_session.h"

#include "dataManager/dataSaver.hpp"
#include "dataManager/dataLoader.hpp"

#include "egoOptimization/anchorOptimize.hpp"

#include <pclomp/ndt_omp.h>
#include <pclomp/gicp_omp.h>

using namespace gtsam;

// 地图优化类
class mapOptimization : public ParamServer {
public:
    // 非线性因子图、初始估计、优化结果等成员变量定义
    gtsam::NonlinearFactorGraph gtSAMgraph;
    gtsam::Values initialEstimate;
    gtsam::Values optimizedEstimate;
    gtsam::ISAM2 *isam;
    gtsam::Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    // 发布器和订阅器，用于ROS消息传递
    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubLaserOdometryGlobal;
    ros::Publisher pubLaserOdometryIncremental;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLocalVertexAndEdge;

    ros::Subscriber subCloud;

    ros::Publisher pubPriorGlobalMap;
    ros::Publisher pubPriorGlobalTrajectory;
    ros::Publisher pubPriorLocalSubmap;
    ros::Publisher pubPriorLocalSubmapCenteriod;

    ros::Subscriber subPose;

    ros::ServiceServer srvSaveMap;  // 保存地图的服务
    ros::ServiceServer srvSaveSession; // 保存会话的服务

    liloc::cloud_info cloudInfo; // 激光云信息结构体

    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames; // 存储表面特征点的关键帧集合
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D; // 3D位姿云
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D; // 6D位姿云
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D; // 复制的3D位姿云
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D; // 复制的6D位姿云

    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // 从odo优化中得到的surf特征集
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // 下采样后的surf特征集

    pcl::PointCloud<PointType>::Ptr laserCloudOri; // 当前扫描得到的激光云
    pcl::PointCloud<PointType>::Ptr coeffSel; // 选择的系数点云

    std::vector<PointType> laserCloudOriSurfVec; // 激光云点的向量，供并行计算使用
    std::vector<PointType> coeffSelSurfVec; // 选择的surf点
    std::vector<bool> laserCloudOriSurfFlag; // 要素标识数组

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer; // 点云地图容器
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap; // 来自地图的surf点云
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS; //来自地图的下采样surf点云

    // Kd树数据结构用于邻域搜索
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses; // 周围关键帧的Kd树
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses; // 历史关键帧的Kd树

    pcl::VoxelGrid<PointType> downSizeFilterSurf; // 点云降采样滤波器
    pcl::VoxelGrid<PointType> downSizeFilterLocalMapSurf; // 本地地图的点云降采样滤波器
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // 用于周边关键帧的降采样

    // 时间相关变量
    ros::Time timeLaserInfoStamp; // 激光信息时间戳
    double timeLaserInfoCur; // 当前激光信息时间

    float transformTobeMapped[6]; // 位姿转换矩阵
    float transformTobeMappedInit[6]; // 初始化时的位姿转换

    std::mutex mtx; // 互斥锁用于多线程安全

    bool isDegenerate = false; // 是否退化标志
    cv::Mat matP; // CV矩阵

    bool systemInitialized = false; // 系统是否初始化
    bool poseInitialized = false; // 位置是否已初始化

    // 文件流与字符串用于保存pose-graph
    std::fstream pgSaveStream; // pg: pose-graph 
    std::vector<std::string> edges_str; // 边的信息
    std::vector<std::string> vertices_str; // 顶点的信息

    int laserCloudSurfFromMapDSNum = 0; // 地图上的DS数量
    int laserCloudSurfLastDSNum = 0; // 上一帧DS数量

    bool aLoopIsClosed = false; // 环闭状态标记

    nav_msgs::Path globalPath; // 全局路径，用于存储位姿信息

    // 仿射变换矩阵
    Eigen::Affine3f transPointAssociateToMap;
    Eigen::Affine3f incrementalOdometryAffineFront;
    Eigen::Affine3f incrementalOdometryAffineBack;

    // 数据管理部分
    std::shared_ptr<dataManager::Session> data_loader = nullptr; // 数据加载器
    std::shared_ptr<dataManager::DataSaver> data_saver = nullptr; // 数据保存器

    // 优化算法部分
    std::unique_ptr<optimization::AnchorOptimization> optimize = nullptr; // 锚点优化器

    pcl::Registration<PointType, PointType>::Ptr registration = nullptr; // 点云注册对象

    // 处理时间统计
    std::vector<double> total_time; // 总时间
    std::vector<double> reg_time; // 注册时间
    std::vector<double> opt_time; // 优化时间

    std::vector<double> ros_time_tum; // ROS TUM时间

public:

    ~mapOptimization() { } // 析构函数

    // 构造函数
    mapOptimization() {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1; // 重线性化阈值
        parameters.relinearizeSkip = 1; // 跳过以执行重线性化的次数
        isam = new ISAM2(parameters); // 创建ISAM2实例

        // 各类发布器的初始化
        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("liloc/mapping/trajectory", 1);
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("liloc/mapping/map_global", 1);
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("liloc/mapping/odometry", 1);
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("liloc/mapping/odometry_incremental", 1);
        pubPath                     = nh.advertise<nav_msgs::Path>("liloc/mapping/path", 1);

        // 订阅来自激光的信息及位置信息
        subCloud = nh.subscribe<liloc::cloud_info>("liloc/deskew/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        subPose  = nh.subscribe("/initialpose", 8, &mapOptimization::initialposeHandler, this, ros::TransportHints().tcpNoDelay());

        // 初始化服务，用于保存地图和会话
        srvSaveMap  = nh.advertiseService("liloc/save_map", &mapOptimization::saveMapService, this);
        srvSaveSession = nh.advertiseService("liloc/save_session", &mapOptimization::saveSessionService, this);

        // 发布最近关键帧的相关信息
        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("liloc/mapping/map_local", 1);
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("liloc/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("liloc/mapping/cloud_registered_raw", 1);

        // 发布全局先验信息
        pubPriorGlobalMap                 = nh.advertise<sensor_msgs::PointCloud2>("liloc/prior/map_prior", 1);
        pubPriorGlobalTrajectory          = nh.advertise<sensor_msgs::PointCloud2>("liloc/prior/traj_prior", 1);
        pubPriorLocalSubmap               = nh.advertise<sensor_msgs::PointCloud2>("liloc/prior/submap_prior", 1);
        pubPriorLocalSubmapCenteriod      = nh.advertise<sensor_msgs::PointCloud2>("liloc/prior/subcenter_prior", 1);
        pubLocalVertexAndEdge             = nh.advertise<visualization_msgs::MarkerArray>("/liloc/prior/local_constrains", 1);

        // 设置降采样滤波器参数
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterLocalMapSurf.setLeafSize(surroundingKeyframeMapLeafSize, surroundingKeyframeMapLeafSize, surroundingKeyframeMapLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // 对于周围关键帧

        initialize(); // 初始化系统
        allocateMemory(); // 分配内存
    }

    void initialize() {
        // 根据不同模式进行相应的数据初始化
        if (mode == ModeType::LIO) {
            // 数据保存器的初始化
            // data_saver.reset(new dataManager::DataSaver(savePCDDirectory, mode));
        }
        else if (mode == ModeType::RELO) {
            // 数据加载器的初始化，会读取G2O文件的数据
            // data_saver.reset(new dataManager::DataSaver(savePCDDirectory, mode));
            data_loader.reset(new dataManager::Session(1, "prior", savePCDDirectory, true));  // FIXME: must use "1"

            pclomp::NormalDistributionsTransform<PointType, PointType>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointType, PointType>());
            pclomp::GeneralizedIterativeClosestPoint<PointType, PointType>::Ptr gicp(new pclomp::GeneralizedIterativeClosestPoint<PointType, PointType>());

            // 配置NDT参数
            ndt->setTransformationEpsilon(ndtEpsilon);
            ndt->setResolution(ndtResolution);

            // 根据设置的方法类型选择对应的注册方法
            if (regMethod == "DIRECT1") {
                ROS_INFO("Using NDT_OMP with DIRECT1.");
                ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
                registration = ndt;
            }
            else if (regMethod == "DIRECT7") {
                ROS_INFO("Using NDT_OMP with DIRECT7.");
                ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
                registration = ndt;
            }
            else if (regMethod == "GICP_OMP") {
                ROS_INFO("Using GICP_OMP.");
                registration = gicp;
            }
            else if (regMethod == "KDTREE") {
                ROS_INFO("Using NDT_OMP with KDTREE.");
                ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
                registration = ndt;
            }
            else {
                ROS_ERROR("Invaild Registration Method !");
                ros::shutdown();
            }

            optimize.reset(new optimization::AnchorOptimization(data_loader, registration)); // 创建优化器
        }
        else {
            ROS_ERROR(" Invaild Mode Type !");
            ros::shutdown(); // 无效模式导致程序关闭
        }
    }

    void allocateMemory() {
        // 分配各种点云内存
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf特征点云
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // 下采样后的surf特征点云

        laserCloudOri.reset(new pcl::PointCloud<PointType>()); // 指定当前激光点云
        coeffSel.reset(new pcl::PointCloud<PointType>()); // 解除选中的对应系数

        // 为surf点分配内存
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false); // 默认所有flag均为false

        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>()); // 创建来自地图的surf点云
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>()); // 创建来自地图的下采样surf点云

        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>()); // 创建kd树

        // 初始化transform数据
        for (int i = 0; i < 6; ++i) {
            transformTobeMapped[i] = 0;
        }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0)); // 初始化协方差矩阵
    }

    void laserCloudInfoHandler(const liloc::cloud_infoConstPtr& msgIn) {
        static double timeLastProcessing = -1; // 上一次处理时间

        timeLaserInfoStamp = msgIn->header.stamp; // 获取当前时间
        timeLaserInfoCur = msgIn->header.stamp.toSec(); // 将ROS时间戳转为秒

        cloudInfo = *msgIn; // 更新激光云信息
        pcl::fromROSMsg(msgIn->cloud_deskewed, *laserCloudSurfLast); // 将ROS格式的点云转换到PCL格式
        
        std::lock_guard<std::mutex> lock(mtx); // 加锁操作，保证线程安全

        if (timeLaserInfoCur - timeLastProcessing < timeInterval) { // 判断时间间隔
            return ; // 小于设定延迟则返回
        }

        // RELO模式下，如果位姿尚未初始化，并且系统也没有完成初始化，则等待initialposeHandler执行给一个初始位置
        if (mode == ModeType::RELO && !poseInitialized) {
            if (!systemInitialized) {
                ROS_WARN("Wait for Initialized Pose ..."); // 系统未初始化警告
                return ;
            }
            else {
                // 初始位姿反变换步骤
                PointTypePose init_pose = trans2PointTypePose(transformTobeMappedInit);

                int submap_id;
                data_loader->searchNearestSubMapAndVertex(init_pose, submap_id); // 寻找最接近的submap及其顶点

                registration->setInputTarget(data_loader->usingSubMap_); // 设置目标点云
                registration->setInputSource(laserCloudSurfLast); // 设置源点云

                // 初始猜测的同质变换
                Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
                Eigen::Matrix3f rotation = eulerToRotation(transformTobeMappedInit[0], transformTobeMappedInit[1], transformTobeMappedInit[2]);
                Eigen::Vector3f translation(transformTobeMappedInit[3], transformTobeMappedInit[4], transformTobeMappedInit[5]);
                init_guess.block(0, 0, 3, 3) = rotation;
                init_guess.block(0, 3, 3, 1) = translation;

                // 执行对齐运算，执行两次，看着是初始位置要给的比较准
                for (int i = 0; i < 2; i++) {
                    CloudPtr aligned(new Cloud());
                    registration->align(*aligned, init_guess); // 执行对齐
                    Eigen::Matrix4f transform;
                    transform = registration->getFinalTransformation(); // 获取最终变换结果
                    init_guess = transform; // 更新新猜测值
                }

                // 后续将获得的变换爆炸解算成欧拉角形式
                Eigen::Vector3f euler = RotMtoEuler(Eigen::Matrix3f(init_guess.block(0, 0, 3, 3)));
                Eigen::Vector3f xyz = init_guess.block(0, 3, 3, 1);

                // 回填新的初始值
                transformTobeMappedInit[0] = euler(0);
                transformTobeMappedInit[1] = euler(1);
                transformTobeMappedInit[2] = euler(2);
                transformTobeMappedInit[3] = xyz(0);
                transformTobeMappedInit[4] = xyz(1);
                transformTobeMappedInit[5] = xyz(2);

                // 标记为已初始化
                systemInitialized = true;
                poseInitialized = true;
            }
        }

        TicToc time; // 计时代码块

        updateInitialGuess(); // 更新初始猜测

        extractSurroundingKeyFrames(); // 提取周围的关键帧

        downsampleCurrentScan(); // 降采样当前扫描获取的点云

        scan2MapOptimization(); // 执行扫描到地图的优化过程

        double t1 = time.toc("1"); // 统计第一阶段时间
        reg_time.push_back(t1); // 记录注册阶段时间

        if (mode == ModeType::RELO) {
            saveRELOKeyFramesAndFactor(); // 如果是RELO模式，则保存关键帧和因子
        }
        else if (mode == ModeType::LIO) {
            saveLIOKeyFramesAndFactor(); // 如果是LIO模式，则保存关键帧和因子
        }
        else {
            ROS_ERROR("Invaild Mode Type. Please use 'LIO' or 'RELO' ... "); // 无效模式错误
            ros::shutdown(); // 关闭ROS系统
        }

        double t2 = time.toc("2"); // 统计第二阶段时间

        opt_time.push_back(t2 - t1); // 记录优化阶段时间

        total_time.push_back(t2); // 记录总阶段时间

        publishOdometry(); // 发布里程计信息

        publishFrames(); // 发布相关框架

        timeLastProcessing = timeLaserInfoCur; // 更新时间最后被处理的时间
    }
    // 将点云数据从局部坐标系转换到地图全局坐标系
    void pointAssociateToMap(PointType const * const pi, PointType * const po) {
        // 使用变换矩阵将输入点 pi 的坐标 (x, y, z) 转换为输出点 po 的 global 坐标
        po->x = transPointAssociateToMap(0, 0) * pi->x + transPointAssociateToMap(0, 1) * pi->y + transPointAssociateToMap(0, 2) * pi->z + transPointAssociateToMap(0, 3);
        po->y = transPointAssociateToMap(1, 0) * pi->x + transPointAssociateToMap(1, 1) * pi->y + transPointAssociateToMap(1, 2) * pi->z + transPointAssociateToMap(1, 3);
        po->z = transPointAssociateToMap(2, 0) * pi->x + transPointAssociateToMap(2, 1) * pi->y + transPointAssociateToMap(2, 2) * pi->z + transPointAssociateToMap(2, 3);
        // 复制强度值
        po->intensity = pi->intensity;
    }

    // 将一个包含平移和旋转信息的数组转换成 gtsam Pose3 对象
    gtsam::Pose3 trans2gtsamPose(float transformIn[]) {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                            gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    // 将 PointTypePose 类型的点转换为 Affine3f（仿射变换）类型
    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint) { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    // 将变换数组转换为 Affine3f（仿射变换）对象
    Eigen::Affine3f trans2Affine3f(float transformIn[]) {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    // 将变换数组转换为 PointTypePose 类型对象
    PointTypePose trans2PointTypePose(float transformIn[]) {
        PointTypePose thisPose6D;
        // 填充 Pose 信息
        thisPose6D.x = transformIn[3]; // 提取 x
        thisPose6D.y = transformIn[4]; // 提取 y
        thisPose6D.z = transformIn[5]; // 提取 z
        thisPose6D.roll  = transformIn[0]; // 提取 roll
        thisPose6D.pitch = transformIn[1]; // 提取 pitch
        thisPose6D.yaw   = transformIn[2]; // 提取 yaw
        return thisPose6D; // 返回封装好的位置姿态
    }

    // 保存地图服务，处理请求的保存map文件操作
    bool saveMapService(liloc::save_mapRequest& req, liloc::save_mapResponse& res) {
        string saveMapDirectory;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;

        // 获取保存目录
        saveMapDirectory = savePCDDirectory;

        std::cout << "Save destination: " << saveMapDirectory << endl;

        // 创建目录并删除旧文件
        int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
        unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());

        // 如果选择了保存路径
        if (req.savepath == 1) {
            pgSaveStream = std::fstream(saveMapDirectory + "/singlesession_posegraph.g2o", std::fstream::out);

            // 保存关键帧变换
            pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
            pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
        
            for(auto& _line: vertices_str)
                pgSaveStream << _line << std::endl; // 保存图的顶点字符串

            for(auto& _line: edges_str)
                pgSaveStream << _line << std::endl; // 保存图的边缘字符串

            pgSaveStream.close(); // 关闭文件流

            // 为 KITTI 和 TUM 格式准备数据流
            std::fstream kittiStream(saveMapDirectory + "/odom_kitti.txt", std::fstream::out);
            std::fstream tumStream(saveMapDirectory + "/odom_tum.txt", std::fstream::out);

            kittiStream.precision(15); // 设置精度
            tumStream.precision(15);

            // 遍历保存数据
            for (size_t i = 0; i < cloudKeyPoses6D->size(); i++) {
                PointTypePose pose = cloudKeyPoses6D->points[i];

                Eigen::Matrix3f rot = eulerToRotation(pose.roll, pose.pitch, pose.yaw);

                // 存储 KITTI 格式的数据
                kittiStream << rot(0, 0) << " " << rot(0, 1) << " " << rot(0, 2) << " " << pose.x << " "
                            << rot(1, 0) << " " << rot(1, 1) << " " << rot(1, 2) << " " << pose.y << " "
                            << rot(2, 0) << " " << rot(2, 1) << " " << rot(2, 2) << " " << pose.z << std::endl;
                
                // 转换至四元数以便存储在 TUM 格式
                Eigen::Matrix3d rot_d = rot.cast<double>();
                Eigen::Vector4d quat = rotationToQuaternion(rot_d);

                tumStream << ros_time_tum[i] << " " << pose.x << " " << pose.y << " " << pose.z << " "
                          << quat(0) << " " << quat(1) << " " << quat(2) << " " << quat(3) << std::endl;
            }

            // 完成时间统计文件的写入
            std::fstream totalTime(saveMapDirectory + "/total_time.txt", std::fstream::out);
            totalTime.precision(5);
            for (auto t : total_time) {
                totalTime << t << std::endl; // 保存总消耗时间
            }
            totalTime.close();

            std::fstream regTime(saveMapDirectory + "/reg_time.txt", std::fstream::out);
            regTime.precision(5);
            for (auto t : reg_time) {
                regTime << t << std::endl; // 保存配准时间
            }
            regTime.close();

            std::fstream optTime(saveMapDirectory + "/opt_time.txt", std::fstream::out);
            optTime.precision(5);
            for (auto t : opt_time) {
                optTime << t << std::endl; // 保存优化时间
            }
            optTime.close();
        }

        // 如果选择了保存点云
        if (req.savecloud == 1) {
            std::string savePcdDirectory = saveMapDirectory + "/PCDs";
            unused = system((std::string("mkdir -p ") + savePcdDirectory).c_str()); // 创建子目录

            pcl::PointCloud<PointType>::Ptr surfCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr surfCloudDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalMapCloudDS(new pcl::PointCloud<PointType>());

            if (req.resolution != 0) {
                cout << "\n\nSave resolution: " << req.resolution << endl;
                downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution); // 设置滤波器分辨率
            }

            // 遍历每个关键帧
            for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
                surfCloud = surfCloudKeyFrames[i]; // 提取当前关键帧

                if (req.resolution != 0) {
                    downSizeFilterSurf.setInputCloud(surfCloud);
                    downSizeFilterSurf.filter(*surfCloudDS); // 降采样
                }

                // 保存下采样后的点云
                std::string name_i = std::to_string(i) + ".pcd"; // 唯一命名
                pcl::io::savePCDFileBinary(savePcdDirectory + "/" + name_i, *surfCloudDS);

                // 全局地图累加更新
                *globalMapCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]); 
            }
            cout << "Processing feature cloud: " << cloudKeyPoses6D->size() << endl;
        
            if (req.resolution != 0) {
                downSizeFilterSurf.setInputCloud(globalMapCloud);
                downSizeFilterSurf.filter(*globalMapCloudDS); // 再次降采样全局地图
            }
        
            // 保存全局地图
            pcl::io::savePCDFileBinary(saveMapDirectory + "/globalMap.pcd", *globalMapCloudDS);
        }

        int ret = 1;

        // 返回成功状态
        res.success = (ret == 1);

        // 重置滤波器叶片大小
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed\n" << endl;

        return true; // 成功完成保存操作
    }

    bool saveSessionService(liloc::save_sessionRequest& req, liloc::save_sessionResponse& res) {
        // 函数功能：保存会话数据，包括位姿图和点云文件。
        // 输入参数：
        // - req: 请求体，其中包含与分辨率相关的信息。
        // - res: 响应体，用来返回保存结果。
        
        if (mode != ModeType::RELO) {
            ROS_ERROR("Not RELO Mode, Can't Save Session !");
            return false;  // 如果当前模式不是RELO，无法保存会话数据。
        }

        std::string saveSessionDir;  // 定义保存目录

        cout << "****************************************************" << endl;
        cout << "Saving session to pcd files ..." << endl;

        saveSessionDir = saveSessionDirectory;  // 获取保存目录

        std::cout << "Save destination: " << saveSessionDir << endl;

        int unused = system((std::string("exec rm -r ") + saveSessionDir).c_str());  
        unused = system((std::string("mkdir -p ") + saveSessionDir).c_str());  // 清空并创建新的会话保存目录
        
        std::string savePcdDirectory = saveSessionDir + "/PCDs"; // 创建PCD文件保存的子目录
        unused = system((std::string("mkdir -p ") + savePcdDirectory).c_str());
        
        pgSaveStream = std::fstream(saveSessionDir + "/singlesession_posegraph.g2o", std::fstream::out); // 打开存储位置图文件的流
            
        pcl::io::savePCDFileBinary(saveSessionDir + "/transformations.pcd", *data_loader->KeyPoses6D_); // 保存变换信息

        for (size_t i = 0; i < data_loader->KeyPoses6D_->size(); i++) {  // 遍历所有关键帧
            if (i == 0) {
                writeVertex(0, pclPointTogtsamPose3(data_loader->KeyPoses6D_->points[i]), vertices_str); // 写入第一个顶点
            }
            else {
                gtsam::Pose3 poseFrom = pclPointTogtsamPose3(data_loader->KeyPoses6D_->points[i - 1]); // 上一帧位态
                gtsam::Pose3 poseTo = pclPointTogtsamPose3(data_loader->KeyPoses6D_->points[i]); // 当前帧位态
                
                gtsam::Pose3 poseRel = poseFrom.between(poseTo); // 计算相对位移

                writeVertex(i, poseTo, vertices_str); // 写入当前顶点
                writeEdge({i - 1, i}, poseRel, edges_str); // 写入边
            }
        }

        for(auto& _line: vertices_str)
            pgSaveStream << _line << std::endl;  // 存储顶点信息

        for(auto& _line: edges_str)
            pgSaveStream << _line << std::endl;  // 存储边信息

        pcl::PointCloud<PointType>::Ptr surfCloud(new pcl::PointCloud<PointType>());  
        pcl::PointCloud<PointType>::Ptr surfCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloudDS(new pcl::PointCloud<PointType>());

        if (req.resolution != 0) { // 检查请求中是否有分辨率设置
            cout << "\n\nSave resolution: " << req.resolution << endl;
            downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution); // 设置降采样的大小
        }

        for (size_t i = 0; i < data_loader->KeyPoses6D_->size(); i++) {
            surfCloud = data_loader->keyCloudVec_[i]; // 获取当前关键帧的点云

            if (req.resolution != 0) {
                downSizeFilterSurf.setInputCloud(surfCloud);
                downSizeFilterSurf.filter(*surfCloudDS); // 对点云进行降采样
            }

            std::string name_i = std::to_string(i) + ".pcd";
            pcl::io::savePCDFileBinary(savePcdDirectory + "/" + name_i, *surfCloudDS); // 保存点云为PCD文件

            *globalMapCloud   += *transformPointCloud(surfCloud, &data_loader->KeyPoses6D_->points[i]); // 构建全局地图
        }
        cout << "Processing feature cloud: " << data_loader->KeyPoses6D_->size() << endl;

        if (req.resolution != 0) {
            downSizeFilterSurf.setInputCloud(globalMapCloud);
            downSizeFilterSurf.filter(*globalMapCloudDS); // 对全局地图进行降采样
        }

        pcl::io::savePCDFileBinary(saveSessionDir + "/globalMap.pcd", *globalMapCloudDS); // 保存全局地图

        int ret = 1;

        res.success = (ret == 1); // 设置响应成功标志

        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize); // 重置降采样大小 

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed\n" << endl;

        pgSaveStream.close(); // 关闭流

        return true; // 返回成功状态
    }

    void visualizeGlobalMapThread() {
        // 函数功能：不断循环以发布全局地图，可在ROS中显示。
        ros::Rate rate(1); 
        while (ros::ok()){ // 循环直到ROS节点被关闭
            rate.sleep();
            publishGlobalMap(); // 发布全局地图
        }
    }

    void drawLinePlot(const std::vector<double>& vec1, const std::vector<double>& vec2, const std::vector<double>& vec3, const std::string& windowName) {
        // 函数功能：绘制折线图展示三个向量的数据。
        // 输入参数：
        // - vec1, vec2, vec3: 要绘制的数据向量
        // - windowName: 窗口名称
        
        int maxLen = std::max({vec1.size(), vec2.size(), vec3.size()}); // 计算最大的长度

        double width = 800, height = 600; // 图像宽高
        cv::Mat plotImg(height, width, CV_8UC3, cv::Scalar(255, 255, 255)); // 初始化白色背景图像

        double margin = 50; // 边距
        double maxVal = 200; // Y轴最大值

        // 绘制X轴和Y轴
        cv::line(plotImg, cv::Point(margin, height - margin), cv::Point(width - margin, height - margin), cv::Scalar(0, 0, 0), 2); 
        cv::line(plotImg, cv::Point(margin, margin), cv::Point(margin, height - margin), cv::Scalar(0, 0, 0), 2);

        // 添加坐标轴标签
        cv::putText(plotImg, "Index", cv::Point(width / 2, height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        cv::putText(plotImg, "ms", cv::Point(10, margin / 2), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

        // 绘制水平网格
        for (int i = 0; i <= maxVal; i += 10) {
            double y = height - margin - (i * (height - 2 * margin) / maxVal);
            cv::line(plotImg, cv::Point(margin, y), cv::Point(width - margin, y), cv::Scalar(200, 200, 200), 1);
            cv::putText(plotImg, std::to_string(i), cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        }

        // 用于绘制每条线的Lambda函数
        auto drawLine = [&](const std::vector<double>& vec, const cv::Scalar& color, const std::string& label, int labelPos) {
            for (size_t i = 1; i < vec.size(); ++i) {
                double x1 = margin + (i - 1) * (width - 2 * margin) / maxLen; // X坐标
                double y1 = height - margin - (vec[i - 1] * (height - 2 * margin) / maxVal); // Y坐标
                double x2 = margin + i * (width - 2 * margin) / maxLen; // 下一点的X坐标
                double y2 = height - margin - (vec[i] * (height - 2 * margin) / maxVal); // 下一点的Y坐标
                cv::line(plotImg, cv::Point(x1, y1), cv::Point(x2, y2), color, 2); // 绘制线段
            }

            cv::putText(plotImg, label, cv::Point(width - margin + 10, labelPos), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2); // 添加标签
        };

        // 绘制三条线
        drawLine(vec1, cv::Scalar(255, 0, 0), "Total Time", 100);
        drawLine(vec2, cv::Scalar(0, 255, 0), "LM Time", 130);
        drawLine(vec3, cv::Scalar(0, 0, 255), "EFGO Time", 160);

        cv::imshow(windowName, plotImg);  // 显示窗口
        cv::waitKey(1); // 等待1毫秒以更新图形
    }

    void displayTime() {
        // 函数功能：持续显示处理时间的折线图
        cv::namedWindow("Processing Times", cv::WINDOW_AUTOSIZE); // 创建名为"Processing Times"的窗口
        
        ros::Rate rate(1);
        while (ros::ok()) {
            rate.sleep();
            // 调用画图函数
            drawLinePlot(total_time, reg_time, opt_time, "Processing Times");
        }
    }
    // 发布全局地图的函数
    void publishGlobalMap() {
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;
        // 设置体素网格滤波器的叶子大小，用于全局地图可视化
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); 

        // 如果模式为重定位（RELO）
        if (mode == ModeType::RELO) {
            // 发布先前的全局地图、关键位姿和使用的子图等信息
            publishCloud(pubPriorGlobalMap, data_loader->globalMap_, timeLaserInfoStamp, mapFrame);
            publishCloud(pubPriorGlobalTrajectory, data_loader->KeyPoses6D_, timeLaserInfoStamp, mapFrame);
            publishCloud(pubPriorLocalSubmap, data_loader->usingSubMap_, timeLaserInfoStamp, mapFrame);
            publishCloud(pubPriorLocalSubmapCenteriod, data_loader->SubMapCenteriod_, timeLaserInfoStamp, mapFrame);
        
            visualizeLocalVertexAndEdge(); // 可视化局部顶点和边
        }
        
        // 检查是否有订阅者
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        // 检查云点集是否为空
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        // 创建KD树用于全局地图
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        std::vector<int> pointSearchIndGlobalMap; // 存储搜索到的点索引
        std::vector<float> pointSearchSqDisGlobalMap; // 存储搜索到的点距离

        mtx.lock(); // 锁定互斥量以保护共享数据
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D); // 设置输入云点
        // 在给定半径内进行搜索
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock(); // 解锁互斥量

        // 将找到的点添加到全局地图关键位姿中
        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);

        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;
        // 设置体素网格滤波器的叶子大小，用于全局地图可视化
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); 
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses); // 输入全局地图关键位姿
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS); // 过滤

        // 为每个下采样后的点设置强度值
        for(auto& pt : globalMapKeyPosesDS->points) {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity; // 获取最近邻的强度值
        }

        // 遍历下采样后的全局地图关键位姿
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i) {
            // 如果与最后一个点的距离大于设定的搜索半径，则跳过
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity; // 获取当前关键帧的索引
            // 转换并累加关键帧的点云
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        }

        // 对全局地图关键帧进行下采样
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS); // 过滤
        // 发布周围激光云
        publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }

    // 初始位姿处理函数
    void initialposeHandler(const geometry_msgs::PoseWithCovarianceStampedConstPtr& poseMsg) {
        ROS_INFO("Initial pose received ."); // 打印接收到初始位姿的信息

        const auto& p = poseMsg->pose.pose.position; // 提取位置
        const auto& q = poseMsg->pose.pose.orientation; // 提取方向

        Eigen::Matrix3d rot = quaternionToRotation(Eigen::Vector4d(q.x, q.y, q.z, q.w)); // 从四元数转换为旋转矩阵
        Eigen::Vector3d euler = RotMtoEuler(rot); // 从旋转矩阵转换为欧拉角

        // 初始化待映射变换
        transformTobeMappedInit[0] = 0.0;
        transformTobeMappedInit[1] = 0.0;
        transformTobeMappedInit[2] = euler(2); // yaw
        transformTobeMappedInit[3] = p.x; // x
        transformTobeMappedInit[4] = p.y; // y
        transformTobeMappedInit[5] = 0.0; // z

        std::cout << "position: " << " ( " << transformTobeMappedInit[3] << ", " << transformTobeMappedInit[4] << ", " << transformTobeMappedInit[5] << " )" << std::endl;

        systemInitialized = true; // 系统初始化标志置为真
    }

    // 可视化局部顶点和边的函数
    void visualizeLocalVertexAndEdge() {
        if (data_loader->usingVertexes_->size() == 0) { // 如果没有顶点，直接返回
            return ;
        }

        visualization_msgs::MarkerArray markerArray; // 创建标记数组

        // 循环节点
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame; // 设置坐标框架
        markerNode.header.stamp = timeLaserInfoStamp; // 设置时间戳
        markerNode.action = visualization_msgs::Marker::ADD; // 添加动作
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST; // 节点类型为球体列表
        markerNode.ns = "vertex"; // 命名空间
        markerNode.id = 0; // ID
        markerNode.pose.orientation.w = 1; // 默认朝向
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; // 尺寸
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1; // 颜色
        markerNode.color.a = 1; // 不透明度

        // 循环边
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame; // 设置坐标框架
        markerEdge.header.stamp = timeLaserInfoStamp; // 设置时间戳
        markerEdge.action = visualization_msgs::Marker::ADD; // 添加动作
        markerEdge.type = visualization_msgs::Marker::LINE_LIST; // 边类型为线段列表
        markerEdge.ns = "edge"; // 命名空间
        markerEdge.id = 1; // ID
        markerEdge.pose.orientation.w = 1; // 默认朝向
        markerEdge.scale.x = 0.1; // 边的宽度
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0; // 颜色
        markerEdge.color.a = 1; // 不透明度

        PointTypePose cur_pose = trans2PointTypePose(transformTobeMapped); // 当前位姿转换为PointTypePose格式

        // 遍历所有使用的顶点
        for (int i = 0; i < data_loader->usingVertexes_->size(); i++) {
            geometry_msgs::Point p;
            p.x = cur_pose.x; // 当前位姿x
            p.y = cur_pose.y; // 当前位姿y
            p.z = cur_pose.z; // 当前位姿z
            markerNode.points.push_back(p); // 添加当前位姿到节点
            markerEdge.points.push_back(p); // 添加当前位姿到边

            p.x = data_loader->usingVertexes_->points[i].x; // 顶点x
            p.y = data_loader->usingVertexes_->points[i].y; // 顶点y
            p.z = data_loader->usingVertexes_->points[i].z; // 顶点z
            markerNode.points.push_back(p); // 添加顶点到节点
            markerEdge.points.push_back(p); // 添加顶点到边
        }

        markerArray.markers.push_back(markerNode); // 将节点添加到标记数组
        markerArray.markers.push_back(markerEdge); // 将边添加到标记数组
        pubLocalVertexAndEdge.publish(markerArray); // 发布标记数组
    }

    // 更新初始猜测的函数
    void updateInitialGuess() {
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped); // 将待映射变换转换为仿射变换

        static Eigen::Affine3f lastImuTransformation; // 保存上一次IMU变换
        if (cloudKeyPoses3D->points.empty()) { // 如果关键点云为空
            if (mode == ModeType::LIO) { // 如果模式为LIO
                transformTobeMapped[0] = cloudInfo.imuRollInit; // 设置roll
                transformTobeMapped[1] = cloudInfo.imuPitchInit; // 设置pitch
                transformTobeMapped[2] = cloudInfo.imuYawInit; // 设置yaw

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // 保存IMU变换
            }
            else if (mode == ModeType::RELO) { // 如果模式为RELO
                Eigen::Affine3f transImuInit = pcl::getTransformation(0, 0, 0, cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw); // 获取IMU初始变换
                Eigen::Affine3f transCbInit = pcl::getTransformation(transformTobeMappedInit[3], transformTobeMappedInit[4], transformTobeMappedInit[5], 
                                                                     transformTobeMappedInit[0], transformTobeMappedInit[1], transformTobeMappedInit[2]); // 获取CB初始变换
                
                Eigen::Affine3f transInit = transCbInit * transImuInit; // 合成初始变换
                pcl::getTranslationAndEulerAngles(transInit, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]); // 提取平移和欧拉角

                lastImuTransformation = pcl::getTransformation(0, 0, 0, transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]); // 保存IMU变换
            }
            else {
                ROS_ERROR("Ivaild mode type !"); // 输出错误信息
                ros::shutdown(); // 关闭ROS
            }
            
            return; // 返回
        }

        // 使用IMU预积分估计进行位姿猜测
        static bool lastImuPreTransAvailable = false; // 上一次IMU预变换是否可用
        static Eigen::Affine3f lastImuPreTransformation; // 上一次IMU预变换
        if (cloudInfo.odomAvailable == true) { // 如果里程计可用
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ, 
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw); // 获取后退变换
            if (lastImuPreTransAvailable == false) { // 如果上一次IMU预变换不可用
                lastImuPreTransformation = transBack; // 保存当前变换
                lastImuPreTransAvailable = true; // 标记为可用
            } 
            else {
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack; // 计算增量变换
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped); // 获取待映射变换
                Eigen::Affine3f transFinal = transTobe * transIncre; // 合成最终变换
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]); // 提取平移和欧拉角

                lastImuPreTransformation = transBack; // 更新上一次IMU预变换

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // 保存IMU变换
                return; // 返回
            }
        }

        // 使用IMU增量估计进行位姿猜测（仅旋转）
        if (cloudInfo.imuAvailable == true && imuType)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // 获取后退变换
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack; // 计算增量变换

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped); // 获取待映射变换
            Eigen::Affine3f transFinal = transTobe * transIncre; // 合成最终变换
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]); // 提取平移和欧拉角

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // 保存IMU变换
            return; // 返回
        }
    }

    // 提取附近关键帧的函数
    void extractNearby() {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>()); // 创建存储附近关键位姿的点云
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>()); // 创建下采样后的点云
        std::vector<int> pointSearchInd; // 存储搜索到的点索引
        std::vector<float> pointSearchSqDis; // 存储搜索到的点距离

        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // 设置KD树输入
        // 在给定半径内进行搜索
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i]; // 获取索引
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]); // 添加到附近关键位姿点云
        }

        // 下采样附近关键位姿
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS); // 过滤
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis); // 最近邻搜索
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity; // 设置强度值
        }

        int numPoses = cloudKeyPoses3D->size(); // 获取关键位姿数量
        for (int i = numPoses-1; i >= 0; --i)
        {
            // 如果当前时间与关键位姿时间差小于5秒，则添加到附近关键位姿
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 5.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break; // 否则退出循环
        }

        extractCloud(surroundingKeyPosesDS); // 提取云点
    }

    // 提取指定点云的函数
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract) {
        laserCloudSurfFromMap->clear(); // 清空提取的点云
        for (int i = 0; i < (int)cloudToExtract->size(); ++i) {
            // 如果与最后一个点的距离大于设定的搜索半径，则跳过
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            int thisKeyInd = (int)cloudToExtract->points[i].intensity; // 获取当前关键帧的索引
            // 如果该索引在地图容器中存在
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) {
                *laserCloudSurfFromMap += laserCloudMapContainer[thisKeyInd].second; // 累加对应的点云
            } 
            else {
                pcl::PointCloud<PointType> laserCloudCornerTemp; // 临时点云
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]); // 转换点云
                *laserCloudSurfFromMap   += laserCloudSurfTemp; // 累加
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp); // 存入地图容器
            }
        }

        // 对提取的点云进行下采样
        downSizeFilterLocalMapSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterLocalMapSurf.filter(*laserCloudSurfFromMapDS); // 过滤
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size(); // 更新下采样后的点云数量

        // 如果地图缓存太大，则清空
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    // 提取周围关键帧的函数
    void extractSurroundingKeyFrames() {
        if (cloudKeyPoses3D->points.empty() == true) // 如果关键点云为空，直接返回
            return; 

        extractNearby(); // 提取附近关键帧
    }

    // 下采样当前扫描的函数
    void downsampleCurrentScan() {
        laserCloudSurfLastDS->clear(); // 清空上次下采样的点云
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast); // 设置输入点云
        downSizeFilterSurf.filter(*laserCloudSurfLastDS); // 过滤
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size(); // 更新下采样后的点云数量
    }

    // 更新点云关联到地图的变换
    void updatePointAssociateToMap() {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped); // 将待映射变换转换为仿射变换
    }

void surfOptimization() {
    updatePointAssociateToMap(); // 更新与地图关联的点

    #pragma omp parallel for num_threads(numberOfCores) // 并行处理，利用多个核心加速计算
    for (int i = 0; i < laserCloudSurfLastDSNum; i = i + 2) { 
        PointType pointOri, pointSel, coeff; // 定义原始点、选择的点和系数
        std::vector<int> pointSearchInd; // 存放最近邻索引
        std::vector<float> pointSearchSqDis; // 存放最近邻平方距离

        pointOri = laserCloudSurfLastDS->points[i]; // 从已有的数据集中提取点
        pointAssociateToMap(&pointOri, &pointSel); // 将原始点与地图点进行关联
        
        // 在地图中搜索与所选点最接近的5个点
        kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

        Eigen::Matrix<float, 5, 3> matA0; // 创建存储邻域点坐标的矩阵
        Eigen::Matrix<float, 5, 1> matB0; // 创建常量矩阵
        Eigen::Vector3f matX0; // 解向量

        matA0.setZero(); // 初始化为零
        matB0.fill(-1); // 设置为-1
        matX0.setZero(); // 初始化解向量为零

        if (pointSearchSqDis[4] < 1.0) { // 如果第5个近邻点的距离小于阈值
            for (int j = 0; j < 5; j++) { 
                // 遍历并填充matA0矩阵
                matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
            }

            // 使用QR分解求解线性方程组
            matX0 = matA0.colPivHouseholderQr().solve(matB0);

            float pa = matX0(0, 0);
            float pb = matX0(1, 0);
            float pc = matX0(2, 0);
            float pd = 1;

            float ps = sqrt(pa * pa + pb * pb + pc * pc); // 归一化因子
            pa /= ps; pb /= ps; pc /= ps; pd /= ps;

            bool planeValid = true; // 用于检测当前面是否有效
            for (int j = 0; j < 5; j++) { 
                // 验证所有点都在同一个平面上
                if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                         pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                         pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                    planeValid = false; // 标记为无效
                    break; // 跳出循环
                }
            }

            if (planeValid) { // 如果检测到是有效的平面
                float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                        + pointOri.y * pointOri.y + pointOri.z * pointOri.z));

                // 系数赋值
                coeff.x = s * pa;
                coeff.y = s * pb;
                coeff.z = s * pc;
                coeff.intensity = s * pd2;

                if (s > 0.1) { // 如果权重大于设置阈值
                    laserCloudOriSurfVec[i] = pointOri; // 保存选择的点
                    coeffSelSurfVec[i] = coeff; // 保存系数
                    laserCloudOriSurfFlag[i] = true; // 更新标志
                }
            }
        }
    }
}

void combineOptimizationCoeffs() { // 合并优化得到的系数
    for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
        if (laserCloudOriSurfFlag[i] == true){ // 检查标志
            laserCloudOri->push_back(laserCloudOriSurfVec[i]); // 添加到待优化云数据集
            coeffSel->push_back(coeffSelSurfVec[i]); // 添加系数
        }
    }

    // 重置旗标，仅用于后续迭代
    std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
}

bool LMOptimization(int iterCount) { // Levenberg-Marquardt 优化
    // 坐标变换说明 (激光雷达和相机之间)
    
    // lidar <- camera      ---     camera <- lidar
    // x = z                ---     x = y
    // y = x                ---     y = z
    // z = y                ---     z = x
    // roll = yaw           ---     roll = pitch
    // pitch = roll         ---     pitch = yaw
    // yaw = pitch          ---     yaw = roll
    
    // 提取旋转矩阵元素
    float srx = sin(transformTobeMapped[2]);
    float crx = cos(transformTobeMapped[2]);
    float sry = sin(transformTobeMapped[1]);
    float cry = cos(transformTobeMapped[1]);
    float srz = sin(transformTobeMapped[0]);
    float crz = cos(transformTobeMapped[0]);

    int laserCloudSelNum = laserCloudOri->size(); // 获取要素总数
    if (laserCloudSelNum < 50) { // 数据不足返回假
        return false; 
    }

    cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0)); // 储存特征点信息
    cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
    cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
    cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

    PointType pointOri, coeff;

    for (int i = 0; i < laserCloudSelNum; i++) { // For each feature point.
        // lidar -> camera
        pointOri.x = laserCloudOri->points[i].x;
        pointOri.y = laserCloudOri->points[i].y;
        pointOri.z = laserCloudOri->points[i].z;

        // lidar -> camera 的系数
        coeff.x = coeffSel->points[i].x;
        coeff.y = coeffSel->points[i].y;
        coeff.z = coeffSel->points[i].z;
        coeff.intensity = coeffSel->points[i].intensity;

        // 根据公式将坐标转换为相机系下的坐标
        float arx = (-srx * cry * pointOri.x - (srx * sry * srz + crx * crz) * pointOri.y + (crx * srz - srx * sry * crz) * pointOri.z) * coeff.x
                  + (crx * cry * pointOri.x - (srx * crz - crx * sry * srz) * pointOri.y + (crx * sry * crz + srx * srz) * pointOri.z) * coeff.y;

        float ary = (-crx * sry * pointOri.x + crx * cry * srz * pointOri.y + crx * cry * crz * pointOri.z) * coeff.x
                  + (-srx * sry * pointOri.x + srx * sry * srz * pointOri.y + srx * cry * crz * pointOri.z) * coeff.y
                  + (-cry * pointOri.x - sry * srz * pointOri.y - sry * crz * pointOri.z) * coeff.z;

        float arz = ((crx * sry * crz + srx * srz) * pointOri.y + (srx * crz - crx * sry * srz) * pointOri.z) * coeff.x
                  + ((-crx * srz + srx * sry * crz) * pointOri.y + (-srx * sry * srz - crx * crz) * pointOri.z) * coeff.y
                  + (cry * crz * pointOri.y - cry * srz * pointOri.z) * coeff.z;
              
        // camera -> lidar
        matA.at<float>(i, 0) = arz;
        matA.at<float>(i, 1) = ary;
        matA.at<float>(i, 2) = arx;
        matA.at<float>(i, 3) = coeff.x;
        matA.at<float>(i, 4) = coeff.y;
        matA.at<float>(i, 5) = coeff.z;
        matB.at<float>(i, 0) = -coeff.intensity; // 负强度用于最小二乘法
    }

    cv::transpose(matA, matAt); // 矩阵转置
    matAtA = matAt * matA; // 求产品
    matAtB = matAt * matB; // 求右边的结果
    cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR); // QR分解求解线性方程

    if (iterCount == 0) { // 仅在第一次调用时执行初始化
        cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

        cv::eigen(matAtA, matE, matV); // 提取特征值和特征向量
        matV.copyTo(matV2); // 将其保留

        isDegenerate = false; // 是否奇异状态标识
        float eignThre[6] = {100, 100, 100, 100, 100, 100}; // 特征值阈值
        for (int i = 5; i >= 0; i--) {
            if (matE.at<float>(0, i) < eignThre[i]) {
                for (int j = 0; j < 6; j++) {
                    matV2.at<float>(i, j) = 0; // 设置奇异值对应项为0
                }
                isDegenerate = true; // 已经是奇异的
            } else {
                break; // 未达到奇异情况
            }
        }
        matP = matV.inv() * matV2; // 保存奇异情况下的调整矩阵
    }

    if (isDegenerate) { // 若存在奇异问题，则调整解
        cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
        matX.copyTo(matX2);
        matX = matP * matX2; // 应用增益修正
    }

    // 更新变换矩阵
    transformTobeMapped[0] += matX.at<float>(0, 0);
    transformTobeMapped[1] += matX.at<float>(1, 0);
    transformTobeMapped[2] += matX.at<float>(2, 0);
    transformTobeMapped[3] += matX.at<float>(3, 0);
    transformTobeMapped[4] += matX.at<float>(4, 0);
    transformTobeMapped[5] += matX.at<float>(5, 0);

    // 判断收敛条件
    float deltaR = sqrt(pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                        pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                        pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));

    float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) +
                        pow(matX.at<float>(4, 0) * 100, 2) +
                        pow(matX.at<float>(5, 0) * 100, 2));

    // 返回是否满足终止标准
    if (deltaR < 0.05 && deltaT < 0.05) {
        return true;
    }
    return false;
}

// 向扫描映射优化过程
void scan2MapOptimization() {
    if (cloudKeyPoses3D->points.empty()) // 如果未读取姿态数据则退出
        return;

    if (laserCloudSurfLastDSNum > 30) { // 至少需要有一定数量的数据
        kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS); // 设置输入源

        for (int iterCount = 0; iterCount < 20; iterCount++) { // 最大迭代次数限制
            laserCloudOri->clear(); // 清空目标集合
            coeffSel->clear(); // 清空参数保存集合

            surfOptimization(); // 执行表面优化操作

            combineOptimizationCoeffs(); // 合并优化得到的系数

            if (LMOptimization(iterCount) == true) // 查找最佳估计
                break;              
        }

        transformUpdate(); // 更新变换参数
    } 
    else {
        ROS_WARN("Not enough features! Only %d planar features available.", laserCloudSurfLastDSNum); // 报警玄机
    }
}

// 更新变换参数
void transformUpdate() {
    if (cloudInfo.imuAvailable == true && imuType) { // 除非IMU可用，且启用了IMU类型
        if (std::abs(cloudInfo.imuPitchInit) < 1.4) { 
            double imuWeight = imuRPYWeight; // 权重赋值
            tf::Quaternion imuQuaternion; // IMU四元数模型
            tf::Quaternion transformQuaternion; // 当前变换四元数
            double rollMid, pitchMid, yawMid;

            // 插值变化以整合IMU信息
            transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
            imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
            tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
            transformTobeMapped[0] = rollMid;

            transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
            imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
            tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
            transformTobeMapped[1] = pitchMid;
        }
    }

    // 限制变换，以确保它们不会过大
    transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
    transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
    transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

    incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped); // 转换成仿射矩阵
}
// 对更新后的值施加约束
float constraintTransformation(float value, float limit) {
    if (value < -limit)
        value = -limit; // 如果值小于负限制，则设置为负限制
    if (value > limit)
        value = limit; // 相干维护范围限制，若值超过正限制，则设置为正限制

    return value; // 返回最后的约束值
}

// 保存帧信息
bool saveFrame() {
    if (cloudKeyPoses3D->points.empty()) // 确认有效点集是否为空
        return true; // 若有效点集为空，则无需保存帧信息

    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back()); // 获取当前帧的起始变换矩阵
    Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]); // 获取目标帧的最终变换矩阵
    
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal; // 计算两个坐标变换之间的关系

    float x, y, z, roll, pitch, yaw; // 位姿各向量初始化
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw); // 从变换中提取平移和欧拉角信息

    // 验证跟进（添加）阈值，以决定是否生成新的关键帧
    if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
        abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
        abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
        sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
        return false; // 如果旋转和位移均在阈值范围内，则不产生新的关键帧

    return true; // 可以形成新的帧
}

// 添加里程计因子
void addOdomFactor() {
    if (cloudKeyPoses3D->points.empty()) { 
        noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished()); // 创建先验噪声模型
        gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise)); // 为 GTSAM 图添加初始先验
        initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped)); // 插入初始估计

        writeVertex(0, trans2gtsamPose(transformTobeMapped), vertices_str); // 写入顶点文件型式
    }
    else { // 处理普通的增量因子
        noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());

        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back()); // 上一帧位置
        gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped); // 当前帧位置
        gtsam::Pose3 poseRel = poseFrom.between(poseTo); // 计算当前帧与上一帧的相对位置关系

        initialEstimate.insert(cloudKeyPoses3D->size(), poseTo); // 添加当前帧到最大尺寸
        gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseRel, odometryNoise)); // 将增量因子加入GTSAM图中

        writeVertex(cloudKeyPoses3D->size(), poseTo, vertices_str); // 写入当前帧的顶点
        writeEdge({cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size()}, poseRel, edges_str); // 写入边缘信息
    }
}

// 添加 Scan 匹配因子
void addScanMatchingFactor() {
    if (cloudKeyPoses3D->points.empty()) {
        return; // 若无可用帧，返回
    }

    PointTypePose cur_pose = trans2PointTypePose(transformTobeMapped); // 转换当前变换为类型安全的Pose格式

    int submap_id; // 地图 id/satellite map
    data_loader->searchNearestSubMapAndVertex(cur_pose, submap_id); // 搜索与当前帧最近的子地图及其顶点

    if (data_loader->usingVertexes_->size() < 2) {
        return; // 若使用的顶点少于2，不进行匹配
    }

    registration->setInputTarget(data_loader->usingSubMap_); // 设置SCAN匹配的目标子地图
    registration->setInputSource(laserCloudSurfLast); // 设置源点云数据

    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity(); // 初始化猜测为单位矩阵
    Eigen::Matrix3f rotation = eulerToRotation(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]); // 构造旋转矩阵
    Eigen::Vector3f translation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]); // 提取位移分量
    init_guess.block(0, 0, 3, 3) = rotation; // 将旋转部分放入初始化猜测
    init_guess.block(0, 3, 3, 1) = translation; // 将位移部分放入初始化猜测

    CloudPtr aligned(new Cloud()); // 定义经过对齐的新点云对象
    registration->align(*aligned, init_guess); // 执行对齐操作

    Eigen::Matrix4f transform;
    transform = registration->getFinalTransformation(); // 获取最终变换矩阵

    std::cout << std::endl;
    std::cout << init_guess << std::endl; // 输出初始猜测的位置
    std::cout << transform << std::endl; // 输出最新的变换结果
    std::cout << std::endl;

    Eigen::Vector3f euler = RotMtoEuler(Eigen::Matrix3f(transform.block(0, 0, 3, 3))); // 从旋转矩阵获取欧拉角
    Eigen::Vector3f xyz = transform.block(0, 3, 3, 1); // 从变换中提取位移

    noiseModel::Diagonal::shared_ptr matchNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back()); // 最新一帧位姿
    gtsam::Pose3 poseTo(gtsam::Rot3::RzRyRx(euler(0), euler(1), euler(2)), gtsam::Point3(xyz(0), xyz(1), xyz(2))); // 创建当前帧的位姿

    gtsam::Pose3 poseRel = poseFrom.between(poseTo); // 计算之前帧与当前帧的相对位姿

    gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), matchNoise)); // 在GTSAM图中增加相对位姿影响因子
}

// 保存RELOKeyFrames和因素
void saveRELOKeyFramesAndFactor() {

    // if (saveFrame() == false)
    //     return;

    optimize->getCurrentPose(cloudKeyPoses3D->points.size(), trans2PointTypePose(transformTobeMapped), laserCloudSurfLast); // 获取优化后的最新位姿
    optimize->addOdomFactors(); // 添加里程计相关因子

    ros_time_tum.push_back(timeLaserInfoCur); // 时间推送机制建立，将时间戳存储

    PointType thisPose3D; // 三维空间位姿
    PointTypePose thisPose6D; // 六维空间位姿
    Pose3 latestEstimate; // 最新期望的位姿

    int latestId = genGlobalNodeIdx(optimize->session_id, cloudKeyPoses3D->points.size()); // 生成全局节点索引
    latestEstimate = optimize->isamCurrentEstimate_.at<Pose3>(latestId); // 更新对应索引下的位姿

    thisPose3D.x = latestEstimate.translation().x(); // Extract the current position coordinates
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size(); // 为光强属性分配当前关键帧数量作为指标
    cloudKeyPoses3D->push_back(thisPose3D); // 向三维点云中添加当前位姿

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity ;

    thisPose6D.roll  = latestEstimate.rotation().roll(); // 提取当前位姿的滚转
    thisPose6D.pitch = latestEstimate.rotation().pitch(); // 提取当前位姿的俯仰
    thisPose6D.yaw   = latestEstimate.rotation().yaw(); // 提取当前位姿的偏航
    thisPose6D.time = timeLaserInfoCur; // 存储当前激光信息时间戳
    cloudKeyPoses6D->push_back(thisPose6D); // 向六维位姿集合添加新元素

    poseCovariance = optimize->isam_->marginalCovariance(latestId); // 获取并展示最新位姿的协方差

    // 保存接收到的边缘点和点云
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame); // 将激光点云复制至关键帧点云

    // 保存关键框架彩色点云
    surfCloudKeyFrames.push_back(thisSurfKeyFrame); // 将该帧加入历史记录，增强数据模型

    // 画路径运动
    updatePath(thisPose6D); // 将路径更新，并反映在系统上
}

// 保存LIO关键帧和因子
void saveLIOKeyFramesAndFactor() {
    if (saveFrame() == false) // 检查是否成功保存帧
        return;

    ros_time_tum.push_back(timeLaserInfoCur); 
    // 添加新的里程计因子  
    addOdomFactor(); 

    // 更新iSAM实时系统
    isam->update(gtSAMgraph, initialEstimate);
    isam->update(); // 执行一次高效的精确化更新
   
    gtSAMgraph.resize(0); // 清空图以便下一次使用
    initialEstimate.clear(); // 清空初始估计

    // 保存关键位姿
    PointType thisPose3D; // 当前三维位姿定义
    PointTypePose thisPose6D; // 当前六维位姿定义（包含旋转信息）
    Pose3 latestEstimate; // 最新的位姿预测

    isamCurrentEstimate = isam->calculateEstimate(); // 计算当前的位姿估计
    latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1); // 获取最新的位姿估计

    thisPose3D.x = latestEstimate.translation().x(); // 获取x坐标
    thisPose3D.y = latestEstimate.translation().y(); // 获取y坐标
    thisPose3D.z = latestEstimate.translation().z(); // 获取z坐标
    thisPose3D.intensity = cloudKeyPoses3D->size(); // 用作索引，表示当前关键点数量
    cloudKeyPoses3D->push_back(thisPose3D); // 将当前三维位姿添加到关键帧列表中

    thisPose6D.x = thisPose3D.x; // 设置六维位姿的x坐标
    thisPose6D.y = thisPose3D.y; // 设置六维位姿的y坐标
    thisPose6D.z = thisPose3D.z; // 设置六维位姿的z坐标
    thisPose6D.intensity = thisPose3D.intensity; // 六维位姿的强度
    thisPose6D.roll  = latestEstimate.rotation().roll(); // 获取滚转角
    thisPose6D.pitch = latestEstimate.rotation().pitch(); // 获取俯仰角
    thisPose6D.yaw   = latestEstimate.rotation().yaw(); // 获取偏航角
    thisPose6D.time = timeLaserInfoCur; // 当前时间戳
    cloudKeyPoses6D->push_back(thisPose6D); // 添加六维位姿到关键帧列表中

    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1); // 计算并存储位姿协方差

    // 保存变换参数
    transformTobeMapped[0] = latestEstimate.rotation().roll(); // 滚转角
    transformTobeMapped[1] = latestEstimate.rotation().pitch(); // 俯仰角
    transformTobeMapped[2] = latestEstimate.rotation().yaw(); // 偏航角
    transformTobeMapped[3] = latestEstimate.translation().x(); // x坐标
    transformTobeMapped[4] = latestEstimate.translation().y(); // y坐标
    transformTobeMapped[5] = latestEstimate.translation().z(); // z坐标

    // 获取表面点云
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>()); // 创建新的点云对象
    pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame); // 拷贝上一个层面点云数据

    surfCloudKeyFrames.push_back(thisSurfKeyFrame); // 添加到关键框架列表中

    updatePath(thisPose6D); // 更新路径
}

// 修正各姿势
void correctPoses() {
    if (cloudKeyPoses3D->points.empty()) // 如果没有关键点则返回
        return;

    if (aLoopIsClosed == true) { // 如果闭环状态为真
        laserCloudMapContainer.clear(); // 清除地图缓存
        globalPath.poses.clear(); // 清除全局路径
        
        int numPoses = isamCurrentEstimate.size(); // 获取当前位姿数量
        for (int i = 0; i < numPoses; ++i) {
            // 更新每个关键帧的位姿
            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
            cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
            cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
            cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

            updatePath(cloudKeyPoses6D->points[i]); // 更新路径
        }

        aLoopIsClosed = false; // 重置循环未关闭状态
    }
}

// 路径更新
void updatePath(const PointTypePose& pose_in) {
    geometry_msgs::PoseStamped pose_stamped; // 创建ROS消息类型，用于存储位姿信息
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time); // 设置时间戳
    pose_stamped.header.frame_id = odometryFrame; // 设置坐标系ID
    pose_stamped.pose.position.x = pose_in.x; // 设置位置x坐标
    pose_stamped.pose.position.y = pose_in.y; // 设置位置y坐标
    pose_stamped.pose.position.z = pose_in.z; // 设置位置z坐标
    tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw); // 从欧拉角创建四元数
    pose_stamped.pose.orientation.x = q.x(); // 设置四元数x分量
    pose_stamped.pose.orientation.y = q.y(); // 设置四元数y分量
    pose_stamped.pose.orientation.z = q.z(); // 设置四元数z分量
    pose_stamped.pose.orientation.w = q.w(); // 设置四元数w分量

    globalPath.poses.push_back(pose_stamped); // 将新位姿添加到全局路径中
}


    void publishOdometry() {
    // 发布全局里程计数据（用于ROS）
    nav_msgs::Odometry laserOdometryROS;  // 创建一个 Odometry 消息对象
    laserOdometryROS.header.stamp = timeLaserInfoStamp;  // 设置时间戳
    laserOdometryROS.header.frame_id = odometryFrame;  // 设置帧ID
    laserOdometryROS.child_frame_id = "odom_mapping";  // 设置子帧ID
    laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];  // 设置位置的x坐标
    laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];  // 设置位置的y坐标
    laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];  // 设置位置的z坐标
    laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);  // 通过姿态信息设置四元数表示的方向
    pubLaserOdometryGlobal.publish(laserOdometryROS);  // 发布全局里程计消息

    // 发布TF变换信息
    static tf::TransformBroadcaster br;  // 静态的 TransformBroadcaster 用于发送 TF 信息
    // 创建从里程计到激光雷达的数据变换
    tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                  tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
    tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");  // 为变换加上时间戳和参考框架
    br.sendTransform(trans_odom_to_lidar);  // 发送变换信息

    // 发布增量里程计数据（用于ROS）
    static bool lastIncreOdomPubFlag = false;  // 标志位，用于判断是否为首次发布增量里程计
    static nav_msgs::Odometry laserOdomIncremental;  // 增量里程计消息
    static Eigen::Affine3f increOdomAffine;  // 增量里程计在仿射空间中的表示
    
    if (lastIncreOdomPubFlag == false) {  // 首次发布时
        lastIncreOdomPubFlag = true;
        laserOdomIncremental = laserOdometryROS;  // 初始化增量里程计为当前全局里程计
        increOdomAffine = trans2Affine3f(transformTobeMapped);  // 将变换转换为仿射矩阵
    } 
    else {
        // 计算从增量里程计前向后的相对变换并更新增量里的状态
        Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;  // 求逆后进行相乘获得新的变换
        increOdomAffine = increOdomAffine * affineIncre;  // 更新增量里的仿射矩阵
        
        float x, y, z, roll, pitch, yaw; // 提取位移和姿态信息
        pcl::getTranslationAndEulerAngles(increOdomAffine, x, y, z, roll, pitch, yaw);  // 从仿射变换中获取位置和欧拉角

        if (cloudInfo.imuAvailable == true && imuType) {  // 如果IMU传感器可用且使用的是IMU类型
            if (std::abs(cloudInfo.imuPitchInit) < 1.4) {  // 限制俯仰角以防超出范围
                double imuWeight = 0.1;  // IMU 权重
                tf::Quaternion imuQuaternion;  // IMU 四元数
                tf::Quaternion transformQuaternion;  // 当前变换的四元数
                double rollMid, pitchMid, yawMid;

                // 插值平滑滚转角
                transformQuaternion.setRPY(roll, 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                roll = rollMid;  // 更新滚转角

                // 插值平滑俯仰角
                transformQuaternion.setRPY(0, pitch, 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                pitch = pitchMid;  // 更新俯仰角
            }
        }

        // 更新增量里程计消息内容
        laserOdomIncremental.header.stamp = timeLaserInfoStamp;  // 设置时间戳
        laserOdomIncremental.header.frame_id = odometryFrame;  // 设置帧ID
        laserOdomIncremental.child_frame_id = "odom_mapping";  // 设置子帧ID
        laserOdomIncremental.pose.pose.position.x = x;  // 更新位置的x坐标
        laserOdomIncremental.pose.pose.position.y = y;  // 更新位置的y坐标
        laserOdomIncremental.pose.pose.position.z = z;  // 更新位置的z坐标
        laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);  // 更新方向四元数

        // 判断变换是否失效，若是则设置协方差
        if (isDegenerate)
            laserOdomIncremental.pose.covariance[0] = 1;  // 协方差非零表示不可信
        else
            laserOdomIncremental.pose.covariance[0] = 0;  // 信任级别高，协方差置零
    }

    pubLaserOdometryIncremental.publish(laserOdomIncremental);  // 发布增量里程计消息
}

void publishFrames() {
    if (cloudKeyPoses3D->points.empty())  // 若无关键帧点，直接返回
        return;

    // 发布关键帧位置
    publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);

    // 发布周围环境的关键帧
    publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);

    // 发布注册的最近关键帧
    if (pubRecentKeyFrame.getNumSubscribers() != 0) {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());  // 新建输出点云
        PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);  // 将变换转为6D位姿
        *cloudOut += *transformPointCloud(laserCloudSurfLastDS, &thisPose6D);  // 转换最后一次激光扫描的点云
        publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);  // 发布关键帧云数据
    }

    // 发布注册后的高分辨率原始点云
    if (pubCloudRegisteredRaw.getNumSubscribers() != 0) {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());  // 创建输出点云
        pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);  // 将去偏移的ROS消息点云转为PCL格式
        PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);  // 将当前位置变换为6D位姿
        *cloudOut = *transformPointCloud(cloudOut, &thisPose6D);  // 应用该变换
        publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);  // 发布经过处理的点云
    }

    // 发布路径信息
    if (pubPath.getNumSubscribers() != 0) {
        globalPath.header.stamp = timeLaserInfoStamp;  // 设置全球路径的时间戳
        globalPath.header.frame_id = odometryFrame;  // 设置框架ID
        pubPath.publish(globalPath);  // 发布全球路径
    }
}


int main(int argc, char** argv) {
    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");  // 输出地图优化开始的信息
    ros::init(argc, argv, "liloc");  // 初始化 ROS 节点

    mapOptimization MO;  // 创建地图优化类对象
    
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);  // 启动线程来可视化全局地图
    // std::thread visualizeTimeThread(&mapOptimization::displayTime, &MO); // 可选择启用显示时间的线程

    ros::spin();  // 保持节点运行

    visualizeMapThread.join();  // 等待可视化线程完成
    // visualizeTimeThread.join(); // 等待时间显示线程完成

    return 0;  // 程序结束返回0
}
