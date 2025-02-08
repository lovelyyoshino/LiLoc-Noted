#include "utility.h"
#include "dataManager/dataSaver.hpp"

// 类 TransformFusion: 用于激光和 IMU 数据融合，提取综合位姿数据。参考：https://github.com/Guo-ziwei/fusion/blob/master/gtsam_imu/src/imu_preint.cpp
class TransformFusion : public ParamServer {
public:
    std::mutex mtx; // 互斥锁用于保护共享资源

    ros::Subscriber subImuOdometry; // 订阅IMU里程计数据
    ros::Subscriber subLaserOdometry; // 订阅激光里程计数据

    ros::Publisher pubImuOdometry; // 发布融合后的IMU里程计数据
    ros::Publisher pubImuPath; // 发布IMU路径信息

    Eigen::Affine3f lidarOdomAffine; // 激光里程计的变换矩阵
    Eigen::Affine3f imuOdomAffineFront; // 最新的前向IMU里程计的变换矩阵
    Eigen::Affine3f imuOdomAffineBack; // 最新的后向IMU里程计的变换矩阵

    tf::TransformListener tfListener; // 坐标变换监听器
    tf::StampedTransform lidar2Baselink; // 从激光到基础链框架的转换

    double lidarOdomTime = -1; // 上一激光里程计更新时间戳
    deque<nav_msgs::Odometry> imuOdomQueue; // 存储IMU里程计算据队列
  
    // 构造函数，进行初始化操作
    TransformFusion() {
        // 如果激光帧与基本链接框架不匹配，尝试获取它们之间的坐标变换
        if(lidarFrame != baselinkFrame) {
            try {
                tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0));
                tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink);
            }
            catch (tf::TransformException ex) {
                ROS_ERROR("%s",ex.what());
            }
        }

        // 订阅激光和IMU里程计消息，并定义回调处理函数
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("liloc/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
        subImuOdometry   = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &TransformFusion::imuOdometryHandler, this, ros::TransportHints().tcpNoDelay());

        // 初始化发布者
        pubImuOdometry   = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);
        pubImuPath       = nh.advertise<nav_msgs::Path>("liloc/imu/path", 1);
    }

    // 将里程计数据转换为仿射变换矩阵
    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom) {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x; // 提取位置
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;

        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation); // 转换四元数
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw); // 获取滚转角、俯仰角和偏航角
        
        return pcl::getTransformation(x, y, z, roll, pitch, yaw); // 返回变换矩阵
    }

    // 处理来自激光里程计的 odometry 消息
    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg) {
        std::lock_guard<std::mutex> lock(mtx); // 加锁以确保线程安全

        lidarOdomAffine = odom2affine(*odomMsg); // 更新激光里程计变换
        lidarOdomTime = odomMsg->header.stamp.toSec(); // 更新时间戳
    }

    // 处理来自IMU的 odometry 消息
    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg) {
        static tf::TransformBroadcaster tfMap2Odom; // 广播器，用于广播 IMU 到 Odom 的转换
        static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0)); 
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

        std::lock_guard<std::mutex> lock(mtx);

        imuOdomQueue.push_back(*odomMsg); // 将IMU信息入队

        // 获取当前IMU时间戳的最新里程计信息
        if (lidarOdomTime == -1) // 检查是否准备好激光里程计
            return;
        
        // 清理队列中过期的信息
        while (!imuOdomQueue.empty()) {
            if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break; // 找到符合要求的就停止
        }
        
        // 计算IMU在前和后的位姿变化
        Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());
        Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());
        Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack; // 得到增量变换
        Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre; // 更新最后的全局状态
        
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw); // 从仿射变换中分离出平移和旋转
        
        // 更新并发布合成后的激光里程计数据
        nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubImuOdometry.publish(laserOdometry); // 发布更新的里程计
        
        static tf::TransformBroadcaster tfOdom2BaseLink;
        tf::Transform tCur;
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur); // 将姿态信息转换为TF格式
        if(lidarFrame != baselinkFrame) // 若帧不同，则应用坐标转换
            tCur = tCur * lidar2Baselink;
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink); // 广播新的基准坐标

        // 路径管理
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;
        double imuTime = imuOdomQueue.back().header.stamp.toSec();
        if (imuTime - last_path_time > 0.1) { 
            last_path_time = imuTime; // 更新时间标记
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            imuPath.poses.push_back(pose_stamped); // 添加新点到路径
            
            // 移除超时的路径点
            while(!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());

            if (pubImuPath.getNumSubscribers() != 0) { // 如果有订阅者则发布路径
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath.publish(imuPath);
            }
        }
    }
};

// 类 IMUPreintegration: 实现IMU预积分，估算位置等
class IMUPreintegration : public ParamServer {
public:
    std::mutex mtx; // 互斥锁保证多线程安全
    
    ros::Subscriber subImu; // 订阅IMU数据
    ros::Subscriber subOdometry; // 订阅增量里程计数据
    ros::Publisher pubImuOdometry; // 发布IMU里程计数据

    bool systemInitialized = false; // 系统是否已初始化标志

    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise; // 初始位姿噪音模型
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise; // 初始速度噪音模型
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise; // 初始偏置噪音模型
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise; // 修正噪音模型
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2; // 第二种修正噪音模型
    gtsam::Vector noiseModelBetweenBias; // 偏置间的噪声模型

    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_; // 优化的IMU集成器
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_; // 原始IMU集成器

    std::deque<sensor_msgs::Imu> imuQueOpt; // 优化的IMU消息队列
    std::deque<sensor_msgs::Imu> imuQueImu; // 原始IMU消息队列

    gtsam::Pose3 prevPose_; // 前一位姿
    gtsam::Vector3 prevVel_; // 前一速度
    gtsam::NavState prevState_; // 前一导航状态o
    gtsam::imuBias::ConstantBias prevBias_; // 前一点的偏置

    gtsam::NavState prevStateOdom; // 先前的整合导航状态
    gtsam::imuBias::ConstantBias prevBiasOdom; // 先前整合的IMU偏置

    bool doneFirstOpt = false; // 标志第一次优化是否完成
    double lastImuT_imu = -1; // 上一次IMU时间戳
    double lastImuT_opt = -1; // 上次优化时间戳

    gtsam::ISAM2 optimizer; // 在线因子图优化器
    gtsam::NonlinearFactorGraph graphFactors; // 非线性因子图
    gtsam::Values graphValues; // 图中的值

    const double delta_t = 0; // 时间差常量

    int key = 1; // 键索引
    
    // 在IMU中心到激光传感器的映射
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

    // 构造函数，设置订阅者和初始配置
    IMUPreintegration() {
        subImu      = nh.subscribe<sensor_msgs::Imu>  (imuTopic,                   2000, &IMUPreintegration::imuHandler,      this, ros::TransportHints().tcpNoDelay());
        subOdometry = nh.subscribe<nav_msgs::Odometry>("liloc/mapping/odometry_incremental", 5, &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());

        pubImuOdometry = nh.advertise<nav_msgs::Odometry> (odomTopic+"_incremental", 2000);

        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        // 每个噪声协方差参数的设置
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // 加速度白噪声
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // 陀螺仪白噪声
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // 积分位置误差
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished()); // 初始偏置

        // 各种草率噪声标准差（位姿、速度、偏置）
        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); 
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // 绝对速度噪声
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 偏置噪声
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // 最终修正噪声
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // 另一种修正噪声
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished(); 
        
        // 设置IMU预积分测量过程及其相关参数
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);  
    }

    // 重设优化器
    void resetOptimization() {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1; // 调整门限
        optParameters.relinearizeSkip = 1; // 重线性化跳过次数
        optimizer = gtsam::ISAM2(optParameters); // 使用参数创建优化器

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors; // 创建零的新因子图

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues; // 清空上一个图的值
    }

    // 重设参数
    void resetParams() {
        lastImuT_imu = -1; // 重置IMU时间戳
        doneFirstOpt = false; // 重置首次优化完成标志
        systemInitialized = false; // 重置系统初始化标志
    }

    // 处理增量里程计消息的方法
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg) {
        std::lock_guard<std::mutex> lock(mtx); // 确保线程安全

        double currentCorrectionTime = ROS_TIME(odomMsg); // 当前纠正时间

        if (imuQueOpt.empty()) // 集成IMU队列为空，返回
            return;

        // 从里程计中提取位置信息
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false; // 判断是否退化
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z)); // 创建激光位姿对象

        if (systemInitialized == false) { // 如果未初始化
            resetOptimization(); // 重设优化器

            while (!imuQueOpt.empty()) { // 移除小于当前修正时间的所有IMU信息
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t) {
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                    imuQueOpt.pop_front();
                }
                else
                    break; // 队列没问题，则适当移动指针
            }

            prevPose_ = lidarPose.compose(lidar2Imu); // 获取从起始IMU到激光背部的乘积
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise); // 定义优先因子
            graphFactors.add(priorPose); // 添加到因子图中

            prevVel_ = gtsam::Vector3(0, 0, 0); // 初始速度为零
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise); // 定义优先因子
            graphFactors.add(priorVel); // 添加到因子图中

            prevBias_ = gtsam::imuBias::ConstantBias(); // 初始偏置条件
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise); // 添加强精度优先因子
            graphFactors.add(priorBias);

            graphValues.insert(X(0), prevPose_); // 插入初始值
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);

            optimizer.update(graphFactors, graphValues); // 更新优化器使用这些初始值
            graphFactors.resize(0); // 清空因子图
            graphValues.clear(); // 清空任务树

            // 重置集成器
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            
            key = 1; // 键重置表单为1，以跟踪序号
            systemInitialized = true; // 表示系统已成功初始化
            return;
        }

        if (key == 100) { // 当键达到100时执行特定操作
            // 从优化结果中恢复更新噪声
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));

            resetOptimization(); // 执行重置操作

            // 增加刚刚获得的数据优先数量包括共识，引导性能
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);

            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);

            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);

            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);

            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1; // 重置键进入正确循环
        }


        while (!imuQueOpt.empty()) { // 遍历优化项
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);
            if (imuTime < currentCorrectionTime - delta_t) { // 如果时间有效
                double dt = (lastImuT_opt < 0) ? (1.0 / imuRate) : (imuTime - lastImuT_opt); // 根据情况选择dt
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt); // 完成一个完整的IMU量测合成
                
                lastImuT_opt = imuTime; // 更新时间戳
                imuQueOpt.pop_front(); // 把当前IMU数据丢掉
            }
            else
                break; // 一旦找到不再继续

        }

        // 将得到的improvements组合做成可用的结构体：history vectors for every state param
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);

        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias))); // 为偏置变量相应地指定因素

        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu); // 激光 + 因素
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise); // 确保约束并纠正至合适层级决定被清洗的类型
        graphFactors.add(pose_factor);

        // 在IBP的一步，将β应用于脱离样直觉包围中
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        graphValues.insert(X(key), propState_.pose());//符号变量 X(key) 关联起来，以便在优化过程中使用，不需要赋值
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);

        optimizer.update(graphFactors, graphValues); // 更新施工图502d
        optimizer.update();
        graphFactors.resize(0); // 尽量保持下式都能瞬间得体.
        graphValues.clear();

        // 估计具体调用先进的hzh解作为输入供给 hence快乐实现对自己的PR失败感知。
        gtsam::Values result = optimizer.calculateEstimate();
        prevPose_  = result.at<gtsam::Pose3>(X(key)); // 叶子负载最优居住设计
        prevVel_   = result.at<gtsam::Vector3>(V(key)); // 边缘生成可行块的空间比较特点
        prevState_ = gtsam::NavState(prevPose_, prevVel_); // 使顾客觉得前面的费用是值得的客户
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key)); // 满意不过是微小..

        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_); //...
    
        // 故障检测：若速度或偏差较大则重置参数
        if (failureDetection(prevVel_, prevBias_)) {
            resetParams();
            return;
        }

        prevStateOdom = prevState_; // 保存状态
        prevBiasOdom  = prevBias_;

        double lastImuQT = -1; // 上一个imu高照测试
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t) { // 保持对齐
            lastImuQT = ROS_TIME(&imuQueImu.front()); // 指向当前所在记录的xf5所毁
            imuQueImu.pop_front(); // 干脆的空...这个务必简单而无情的放弃！
        }

        if (!imuQueImu.empty()) {
            // 控制IMU积分机利用之前的ODOMBIAS实例
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            for (int i = 0; i < (int)imuQueImu.size(); ++i) { // 逐行扫描显示组
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / imuRate) :(imuTime - lastImuQT); // 区间段才预测结果维护正常

                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z), dt); // 回归一种距离均衡行为的稳定化问题
                lastImuQT = imuTime; // 有效周期末端生成掩模rdaA自动生长传播补丁，限制2D极少量步骤真正在战斗。
            }
        }

        ++key; // 无节制增加输出关键，随即适应策略，不然ORIZONTAL力量突然爆发，只会失去原来的东西。
        doneFirstOpt = true; // 表达接受压力已经运行顺利引深刻影响用户感受反馈行动代言人！又这里体现它能推动对象实施通讯规范方案；
    }

    // 故障检测模块，根据速度和偏倚范围判断
    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur) {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z()); // 获得节点向量
        if (vel.norm() > 30) { // 大于阈值则警告
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true; // 有异常大速率故障将返回
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0) { // 岩Hash确认问题
            ROS_WARN("Large bias, reset IMU-preintegration!");
            return true; // 出现偏离现象则报错
        }

        return false; // 可忽略
    }

    // 处理IMU数据并按照处理请求来控制综合流程
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw) {
        std::lock_guard<std::mutex> lock(mtx); // 防止多线程冲突

        sensor_msgs::Imu thisImu = imuConverter(*imu_raw); // 过滤IMU源头包装时刻信号，这段厚实冻结感觉精神力不断提取吸收。

        if (correct) {
            // 感应合并内容比例缩减
            thisImu.linear_acceleration.x = thisImu.linear_acceleration.x * kAccScale;
            thisImu.linear_acceleration.y = thisImu.linear_acceleration.y * kAccScale;
            thisImu.linear_acceleration.z = thisImu.linear_acceleration.z * kAccScale;
        }

        // 收集和排序IMU标准
        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        if (doneFirstOpt == false) // 如果尚未完成首届优化则返回
            return;

        double imuTime = ROS_TIME(&thisImu); // 当前时间皮吨颗粒度迭降
        double dt = (lastImuT_imu < 0) ? (1.0 / imuRate) : (imuTime - lastImuT_imu); // 被迫选尽质检常规环境
        lastImuT_imu = imuTime; // 更新时间

        // 将当前IMU数据单独消耗
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x, thisImu.angular_velocity.y, thisImu.angular_velocity.z), dt); 

        // 预测里程计，然后信息封装方式缓慢展示
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom); // 整合重复使用调用也就是收敛目标推仍需文良通融改造方法的基本需求！

        // 城市详细信息统计和数据挂牌发布
        nav_msgs::Odometry odometry; // 里程计基本信息线充填
        odometry.header.stamp = thisImu.header.stamp; // 恶劣巩固信号显示项目应用模式服务更多体系参与开发命名规则参照
        odometry.header.frame_id = odometryFrame; // 容许外部呼叫需要明确中的方向配置 格式错误可能造成间接最大损失信号
        odometry.child_frame_id = "odom_imu"; // 然后附加知道这里是什么趋势

        // IMU姿态到激光标定参考
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar); // 测试相对重要延续本质粗黑启动意味着阶段一切实际政策活动致密希望且使得飞行效果说明

        // 编排非碰撞并透露细腻绿色文本 استان سیاه و سفیداندازهای fimmä simitologidanför утро мнения元素 Васне папки взгадчук видcció завал预算布局
        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        
        odometry.twist.twist.linear.x = currentState.velocity().x(); // 时频最近倍设定的对应各类时装派 特烈腐烂程度通过感觉决定准确容易显著权威建议事务上诉长度اتھانا فر نیاز بدانید .
        odometry.twist.twist.linear.y = currentState.velocity().y(); // 套妆翻天叹专熟视觉統計西丽无委屈陌众淡青便一寸走出装备，会暴露一定来源。。其中多少かったりの別ストライン。加碼きかいよは酌量続く。
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x(); // 建議社交推出體驗設核紀念父母興趣讓他們得到滿足該事業組件三科進展前盤擠壓貨幣大佬 引導視野銜鐵馬周面 瑕疵宗教 熟悉長期形成贈品破釋豁然常纖默帖痛苦夢見水果冰隨推詞便衣目間如繁忙形狀尋範尺預畢打開商會磨璃毛或積情原經风格۔

        // 及时发送里程计更新
        pubImuOdometry.publish(odometry);
    }
};

// 主函数
int main(int argc, char** argv) {
    ros::init(argc, argv, "roboat_loam"); // 初始化ROS节点
    
    IMUPreintegration ImuP; // 创建设备有关IMU预积分的主要组件

    TransformFusion TF; // 创建用于融合IMU和激光的数据结构

    ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");
    
    ros::MultiThreadedSpinner spinner(4); // 允许多线程调动环路，快速响应和处理数据
    spinner.spin(); // 开始循环，等待并处理回调调用
    
    return 0;
}
