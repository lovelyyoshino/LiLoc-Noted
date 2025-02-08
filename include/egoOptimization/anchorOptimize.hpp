#pragma once

#ifndef _ANCHOR_OPTIMIZE_HPP_
#define _ANCHOR_OPTIMIZE_HPP_

#include "../utility.h"
#include "../dataManager/dataLoader.hpp"

#include "../tictoc.h"

#include "BetweenFactorWithAnchoring.h"

#include <pclomp/ndt_omp.h>
#include <pclomp/gicp_omp.h>

namespace optimization {

class AnchorOptimization : public ParamServer {
public:
    std::mutex mtx; // 用于多线程操作的互斥锁

    std::shared_ptr<dataManager::Session> priorSession_ = nullptr; // 指向数据管理会话的指针
    pcl::Registration<PointType, PointType>::Ptr registration_ = nullptr; // 点云配准对象指针


    gtsam::NonlinearFactorGraph gtSAMgraph_; // GTSAM非线性因子图
    gtsam::ISAM2 *isam_; // 增量式平衡结构体
    gtsam::Values initialEstimate_; // 初始估计值
    gtsam::Values isamCurrentEstimate_; // 当前估计值

    gtsam::Pose3 poseOrigin_; // 原点位姿

    gtsam::noiseModel::Diagonal::shared_ptr priorNoise_; // 先验噪声模型
    gtsam::noiseModel::Diagonal::shared_ptr odomNoise_; // 里程计噪声模型
    gtsam::noiseModel::Diagonal::shared_ptr matchNoise_; // 匹配噪声模型
    gtsam::noiseModel::Diagonal::shared_ptr largeNoise_; // 大噪声模型
    gtsam::noiseModel::Base::shared_ptr robustNoise_; // 鲁棒噪声模型

    std::vector<int> priorNodePtIds_; // 存储先前节点的点ID
    std::vector<int> increNodePtIds_; // 存储增量节点的点ID
    std::vector<int> currOdomNodeIds_; // 存储当前里程计节点的ID

    int key_; // 当前关键帧标识符
    PointTypePose curPose_; // 当前位姿
    CloudPtr curCloud_; // 当前点云

    const int session_id = 0; // 会话ID

    const int marg_size = 10; // 边际化大小

    bool first_add = true; // 标记是否为第一次添加

    pcl::VoxelGrid<PointType> downSizeFilterSurf; // 点云下采样滤波器

public:
    ~AnchorOptimization() { } // 析构函数
    // 构造函数，初始化成员变量并设置相关参数
    AnchorOptimization(const std::shared_ptr<dataManager::Session>& session, const pcl::Registration<PointType, PointType>::Ptr &reg) 
    : priorSession_(session), registration_(reg)
    { 
        // 将每一个关键位姿的索引加入至priorNodePtIds_中
        for (size_t i = 0; i < priorSession_->KeyPoses6D_->size(); i++) {
            priorNodePtIds_.push_back(i);
        }

        // 初始化原点位姿
        poseOrigin_ = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));

        initOptimizer(); // 初始化优化算法
        initNoiseConstants(); // 初始化噪声常数
        addPriorSessionToGraph(); // 将先前会话添加到因子图中
        optimizeGraph(1); // 优化因子图
        allocateMemory(); // 分配内存

        ROS_INFO_STREAM("\033[1;32m Anchor Optimization is initialized successfully \033[0m"); // 日志信息输出
    }

    // 分配内存给curCloud_
    void allocateMemory() {
        curCloud_.reset(new Cloud());
    }

    // 初始化GTSAM的优化器
    void initOptimizer(){
        gtsam::ISAM2Params parameters; // 定义ISAM2的参数
        parameters.relinearizeThreshold = 0.1; // 重线性化阈值
        parameters.relinearizeSkip = 1; // 重线性化跳过次数
        isam_ = new gtsam::ISAM2(parameters); // 创建新的ISAM2对象
    }

    // 初始化噪声常数，包括 prior、odom、match、large 和 robust 的噪声模型
    void initNoiseConstants() {
        // Variances Vector6 表示 rad*rad, rad*rad, rad*rad, meter*meter, meter*meter, meter*meter 的方差
        {
            gtsam::Vector Vector6(6);
            Vector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12; // 设置非常小的先验噪声
            priorNoise_ = gtsam::noiseModel::Diagonal::Variances(Vector6);
        }

        {
            gtsam::Vector Vector6(6);
            Vector6 << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6; // 设置较小的里程计噪声
            odomNoise_ = gtsam::noiseModel::Diagonal::Variances(Vector6);
        }

        {
            gtsam::Vector Vector6(6);
            Vector6 << 1e-5, 1e-5, 1e-5, 1e-3, 1e-3, 1e-3; // 设置匹配产生的噪声
            matchNoise_ = gtsam::noiseModel::Diagonal::Variances(Vector6);
        }

        {
            gtsam::Vector Vector6(6);
            Vector6 << M_PI * M_PI, M_PI * M_PI, M_PI * M_PI, 1e8, 1e8, 1e8; // 设置大噪声（用于鲁棒处理）
            largeNoise_ = gtsam::noiseModel::Diagonal::Variances(Vector6);
        }

        float robustNoiseScore = 0.5; // 常量设定
        gtsam::Vector robustNoiseVector6(6); 
        robustNoiseVector6 << robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore;
        // 创建鲁棒噪声模型
        robustNoise_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(1), 
            gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6)
        );
    }

    // 将先前会话的信息添加到因子图中
    void addPriorSessionToGraph() {
        int this_session_anchor_node_idx = genAnchorNodeIdx(priorSession_->index_);//这里将会话索引做一个偏移，以便于区分不同会话的锚点，最大是kSessionStartIdxOffset = 1000000

        if (priorSession_->is_base_session_) {//如果是基础会话
            gtSAMgraph_.add(gtsam::PriorFactor<gtsam::Pose3>(this_session_anchor_node_idx, poseOrigin_, priorNoise_));
        }
        else {
            gtSAMgraph_.add(gtsam::PriorFactor<gtsam::Pose3>(this_session_anchor_node_idx, poseOrigin_, largeNoise_));
        }

        initialEstimate_.insert(this_session_anchor_node_idx, poseOrigin_);//设置初始位置值

        // 添加节点 
        for (auto& _node: priorSession_->nodes_) {
            int node_idx = _node.second.idx;
            auto& curr_pose = _node.second.initial;

            int curr_node_global_idx = genGlobalNodeIdx(priorSession_->index_, node_idx);

            if (node_idx == 0) {
                // 先验节点
                gtSAMgraph_.add(gtsam::PriorFactor<gtsam::Pose3>(curr_node_global_idx, curr_pose, priorNoise_));
                initialEstimate_.insert(curr_node_global_idx, curr_pose);
            }
            else {
                // 里程计节点
                gtSAMgraph_.add(gtsam::PriorFactor<gtsam::Pose3>(curr_node_global_idx, curr_pose, priorNoise_));
                initialEstimate_.insert(curr_node_global_idx, curr_pose);
            }
        }

        // 添加边 
        for (auto& _edge: priorSession_->edges_) {
            int from_node_idx = _edge.second.from_idx;
            int to_node_idx = _edge.second.to_idx;

            int from_node_global_idx = genGlobalNodeIdx(priorSession_->index_, from_node_idx);
            int to_node_global_idx = genGlobalNodeIdx(priorSession_->index_, to_node_idx);

            gtsam::Pose3 relative_pose = _edge.second.relative;

            if (std::abs(to_node_idx - from_node_idx) == 1) {
                // 如果相邻节点之间创建约束
                gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(from_node_global_idx, to_node_global_idx, relative_pose, priorNoise_));
            }
        }
    }

    // 优化因子图
    void optimizeGraph(int _inter) {
        isam_->update(gtSAMgraph_, initialEstimate_); // 更新ISAM2

        while (--_inter) {
            isam_->update(); // 多次更新以达到更好的收敛
        }

        isamCurrentEstimate_ = isam_->calculateEstimate(); // 计算当前估计值

        gtSAMgraph_.resize(0); // 清空因子图
        initialEstimate_.clear(); // 清空初始估计值

        updateSessionPoses(); // 更新会话位姿

        ROS_INFO_STREAM("Optimize ... " << " Have prior nodes: " << priorNodePtIds_.size() << ", current odometry nodes: " << currOdomNodeIds_.size() << " and incremental nodes: " << increNodePtIds_.size());
    }

    // 获取当前的位置信息以及对应的云数据
    void getCurrentPose(const int &key, const PointTypePose & cur_pose, const CloudPtr & cur_cloud) {
        curPose_ = cur_pose; // 保存当前位姿
        key_ = key; // 保存关键帧ID
        curCloud_ = cur_cloud; // 保存当前点云
    }

    // 对不再使用的节点进行边际化处理
    void margilization() {
        if ((key_ != 0 && key_ % 10 == 0 || priorSession_->margFlag)) { // 每隔10个关键帧进行边际化或根据标志执行
            int currentId = genGlobalNodeIdx(session_id, key_);
            
            // 获取当前节点及其相关的高斯噪声，isam_->marginalCovariance(currentId): 这是调用 isam_ 对象的 marginalCovariance 方法，并传入 currentId 参数。marginalCovariance 方法通常用于获得某个特定变量的边际协方差矩阵
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(isam_->marginalCovariance(currentId));//获取当前节点的噪声
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(isam_->marginalCovariance(currentId));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(isam_->marginalCovariance(currentId));
        }
    }

    // 添加与运动有关的因素，会先执行getCurrentPose
    void addOdomFactors() {
        TicToc time("1");

        addLidarFactor(); // 添加激光雷达因素

        double t1 = time.toc("lidar factor"); // 记录时间
        
        // std::cout << " lidar factor :" << t1 << std::endl;

        addImuFactor(); // 添加IMU因素

        // 添加扫描匹配因素
        addScanMatchingFactor(); 

        double t2 = time.toc("scan factor"); // 记录时间

        // std::cout << " scan factor :" << t2 - t1 << std::endl;

        optimizeGraph(1); // 执行优化

        double t3 = time.toc("opt"); // 记录时间

        // std::cout << " opt :" << t3 - t2 << std::endl;
    
        margilization(); // 执行边际化，在addScanMatchingFactor就使用了searchNearestSubMapAndVertex来更新margFlag

        double t4 = time.toc("marg"); // 记录时间

        // std::cout << " marg :" << t4 - t3 << std::endl;
    }

    // 添加激光雷达因素
    void addLidarFactor() {
        if (key_ == 0) { // 如果是第一帧，也就是没有数据预先加载或者没有先验数据
            int this_session_anchor_node_idx = genAnchorNodeIdx(session_id);
            gtSAMgraph_.add(gtsam::PriorFactor<gtsam::Pose3>(this_session_anchor_node_idx, poseOrigin_, largeNoise_));
            initialEstimate_.insert(this_session_anchor_node_idx, poseOrigin_);

            int this_node_id = genGlobalNodeIdx(session_id, key_);
            gtSAMgraph_.add(gtsam::PriorFactor<gtsam::Pose3>(this_node_id, pclPointTogtsamPose3(curPose_), priorNoise_));
            initialEstimate_.insert(this_node_id, pclPointTogtsamPose3(curPose_));

            currOdomNodeIds_.push_back(key_); // 加入当前里程计ID
        }
        else {
            int this_node_id = genGlobalNodeIdx(session_id, key_);

            gtsam::Pose3 poseFrom = isamCurrentEstimate_.at<gtsam::Pose3>(this_node_id - 1); // 获取上一个节点的位姿
            gtsam::Pose3 poseTo = pclPointTogtsamPose3(curPose_); // 转换当前位姿
            
            gtsam::Pose3 poseRel = poseFrom.between(poseTo); // 计算相对位姿
            initialEstimate_.insert(this_node_id, poseTo); // 插入当前节点位姿

            gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(this_node_id - 1, this_node_id, poseRel, odomNoise_)); // 添加边
            currOdomNodeIds_.push_back(key_); // 加入当前里程计ID
        }
    }

    // 添加IMU因素，目前该功能未实现。
    void addImuFactor() {
        
    }

void addNewPrior() {
    static PointTypePose lastPose; // 上一个姿态
    static int end = priorSession_->KeyPoses6D_->size(); // 记录之前关键位姿数量的变量
    static int count = 0; // 用于计算添加次数

    if (first_add) { // 如果是第一次添加优先节点
        curPose_.intensity = priorSession_->KeyPoses6D_->size(); // 当前姿态强度设置为当前关键点数量
        priorSession_->KeyPoses6D_->push_back(curPose_); // 添加当前姿态

        CloudPtr copy(new Cloud()); 
        pcl::copyPointCloud(*curCloud_, *copy); // 拷贝当前点云数据到新的拷贝
        priorSession_->keyCloudVec_.push_back(copy); // 将拷贝的点云添加入会话

        int this_node_id = genGlobalNodeIdx(priorSession_->index_, (int)curPose_.intensity); // 生成全局节点索引

        gtsam::Pose3 poseFrom = isamCurrentEstimate_.at<gtsam::Pose3>(this_node_id - 1); // 获取上一个状态估计
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(curPose_); // 将点云转换为GTSAM Pose3格式的目标姿态
        initialEstimate_.insert(this_node_id, poseTo); // 插入初始估计

        gtsam::Pose3 poseRel = poseFrom.between(poseTo); // 计算两个姿态之间的相对偏差

        gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(this_node_id - 1, this_node_id, poseRel, odomNoise_)); // 在图中添加边（因子）

        increNodePtIds_.push_back(curPose_.intensity); // 增加新节点ID到更新列表，这是增量节点的ID列表
        count++; // 更新添加计数
        first_add = false; // 后续添加将不再被视为第一次添加
        lastPose = curPose_; // 更新最后的姿态
    } else { // 如果不是第一次添加，即发现当前点在searchNearestSubMapAndVertex中关键点数量都超过了十个，则重新创建子图
        Eigen::Affine3f transStart = pclPointToAffine3f(lastPose); // 将上一位姿转化为仿射变换
        Eigen::Affine3f transFinal = pclPointToAffine3f(curPose_); // 将当前位姿转化为仿射变换

        Eigen::Affine3f transBetween = transStart.inverse() * transFinal; // 计算两次变换间的相对变换

        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw); // 从变换中提取平移和旋转角度

        if (abs(roll) < surroundingkeyframeAddingAngleThreshold && // 判断与上一帧的姿态变化是否在阈值范围内
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold) 
        {
            return ; // 若变化小，直接返回，不添加新优先节点
        }

        curPose_.intensity = priorSession_->KeyPoses6D_->size(); // 更新当前姿态强度
        priorSession_->KeyPoses6D_->push_back(curPose_); // 添加到关键位姿集合

        CloudPtr copy(new Cloud());
        pcl::copyPointCloud(*curCloud_, *copy); // 拷贝当前点云
        priorSession_->keyCloudVec_.push_back(copy); // 存入会话
        
        int this_node_id = genGlobalNodeIdx(priorSession_->index_, (int)curPose_.intensity); // 生成当前节点ID
        gtsam::Pose3 poseFrom = isamCurrentEstimate_.at<gtsam::Pose3>(this_node_id - 1); // 获取前一节点姿态
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(curPose_); // 转换为GTSAM的Pose3格式
        initialEstimate_.insert(this_node_id, poseTo); // 插入初始估计
       
        gtsam::Pose3 poseRel = poseFrom.between(poseTo); // 计算相对姿态差

        gtSAMgraph_.add(gtsam::BetweenFactor<gtsam::Pose3>(this_node_id - 1, this_node_id, poseRel, odomNoise_)); // 加入图优化过程 

        increNodePtIds_.push_back(curPose_.intensity); // 更新节点ID列表
        count++; // 增加计数器
        first_add = false; // 设置标志以避免再次复用此逻辑
        lastPose = curPose_; // 更新lastPose为当前姿态

        if (count % priorSession_->submap_size == 0) { // 检查统计数量是否达到子地图大小，认为需要10个，然后就生成一个子地图
            CloudPtr submap(new Cloud()); // 创建新的子地图
            TrajectoryPtr vertexCloud(new Trajectory()); // 保存对应顶点集
            
            float xx = 0.0, xy = 0.0, xz = 0.0;

            for (int i = end; i < priorSession_->keyCloudVec_.size(); i++) { // 遍历从end开始的新加入的关键位姿
                *submap += *transformPointCloud(priorSession_->keyCloudVec_[i], &priorSession_->KeyPoses6D_->points[i]); // 变换并合并到子地图
                xx += priorSession_->KeyPoses6D_->points[i].x; // 聚合所有X坐标
                xy += priorSession_->KeyPoses6D_->points[i].y; // 聚合所有Y坐标
                xz += priorSession_->KeyPoses6D_->points[i].z; // 聚合所有Z坐标

                vertexCloud->push_back(priorSession_->KeyPoses6D_->points[i]); // 把当前顶点推入数组
            }

            PointTypePose centeriod; // 中心点实例
            centeriod.x = xx / (float)priorSession_->submap_size; // 计算中心点
            centeriod.y = xy / (float)priorSession_->submap_size;
            centeriod.z = xz / (float)priorSession_->submap_size;
            priorSession_->SubMapCenteriod_->push_back(centeriod); // 存储到子图中心点集合

            priorSession_->subMapVertexCloudVec_.push_back(vertexCloud); // 推入存储结构

            CloudPtr submap_copy(new Cloud());      

            downSizeFilterSurf.setLeafSize(0.2, 0.2, 0.2); // 设置下采样过滤参数
            downSizeFilterSurf.setInputCloud(submap); // 设置输入子图点云
            downSizeFilterSurf.filter(*submap); // 执行过滤操作

            pcl::copyPointCloud(*submap, *submap_copy); // 复制处理后的子图
            priorSession_->subMapCloudVec_.push_back(submap_copy); // 添加到子图集合

            // *priorSession_->globalMap_ += *submap_copy; // 全局地图更新，可选项（注释掉了）

            end = priorSession_->KeyPoses6D_->size(); // 更新结束位置为当前
            count = 0; // 重置计数器
            submap->clear(); // 清空当前子图
        }
    }
}

void addScanMatchingFactor() {
    int submap_id;
    priorSession_->searchNearestSubMapAndVertex(curPose_, submap_id); // 搜索最接近的子图和顶点

    if (priorSession_->usingVertexes_->size() <= 1) { // 如果使用的顶点数量少于等于1，认为这个submap附近没有任何的关键帧
        addNewPrior(); // 添加新的优先节点作为补充，和searchNearestSubMapAndVertex相互补充
        return ;
    }

    first_add = true; // 标记为第一个匹配因素添加

    registration_->setInputTarget(priorSession_->usingSubMap_); // 设置目标点云为使用的子图
    registration_->setInputSource(curCloud_); // 设置源点云为当前点云
    
    Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity(); // 初始化矩阵为单位矩阵
    Eigen::Matrix3f rotation = eulerToRotation(curPose_.roll, curPose_.pitch, curPose_.yaw); // 根据欧拉角计算旋转矩阵
    Eigen::Vector3f translation(curPose_.x, curPose_.y, curPose_.z); // 提取当前位姿的平移分量
    init_guess.block(0, 0, 3, 3) = rotation; // 设置旋转部分
    init_guess.block(0, 3, 3, 1) = translation; // 设置平移部分
    
    CloudPtr aligned(new Cloud()); 
    registration_->align(*aligned, init_guess); // 对齐当前点云以及初始化猜测

    Eigen::Matrix4f transform; 
    transform = registration_->getFinalTransformation(); // 获取最终变换结果

    Eigen::Vector3f euler = RotMtoEuler(Eigen::Matrix3f(transform.block(0, 0, 3, 3))); // 将旋转矩阵转换为欧拉角用于后续运算
    Eigen::Vector3f xyz = transform.block(0, 3, 3, 1); // 提取变换后的平移向量

    int this_node_id = genGlobalNodeIdx(session_id, key_); // 生成当前节点ID

    gtsam::Pose3 poseTo(gtsam::Rot3::RzRyRx(euler(0), euler(1), euler(2)), gtsam::Point3(xyz(0), xyz(1), xyz(2))); // 定义目标姿态

    if (key_ != 0) { // 如果不是初始关键帧
        gtsam::Pose3 poseFrom = isamCurrentEstimate_.at<gtsam::Pose3>(this_node_id - 1); // 获取上一个节点的姿态
        gtsam::Pose3 poseRel = poseFrom.between(poseTo); // 计算相对偏转

        gtSAMgraph_.add(BetweenFactor<Pose3>(this_node_id - 1, this_node_id, poseFrom.between(poseTo), matchNoise_)); // 向图中添加约束
    }

    CloudPtr curCloud_trans(new Cloud());  
    PointTypePose pose_match = gtsamPose3ToPclPoint(poseTo); // 将poseTo转为点云格式
    *curCloud_trans += *transformPointCloud(curCloud_, &pose_match); // 应用变换至当前点云

    for (auto & id : priorSession_->localGoup_) { 
        PointTypePose pose_i = priorSession_->subMapVertexCloudVec_[submap_id]->points[id]; // 获取子图中的每个位姿点
        int prior_id = priorSession_->subMapVertexCloudVec_[submap_id]->points[id].intensity; // 强度用作 ID

        auto it = std::find(increNodePtIds_.begin(), increNodePtIds_.end(), prior_id);
        if (it != increNodePtIds_.end()) { // 如果该 ID 已经存在跳过
            continue ;
        }

        CloudPtr cloud_i(new Cloud()); 
        *cloud_i += *priorSession_->keyCloudVec_[prior_id]; // 获取对应的关键点云

        CloudPtr cloud_i_trans(new Cloud());
        *cloud_i_trans += *transformPointCloud(cloud_i, &pose_i); // 进行变换得到云

        float score = getICPFitnessScore(cloud_i_trans, curCloud_trans); // 得分计算
        if (score > 2) { // 若得分超过最大限制则修正
            score = 2; 
        }

        gtsam::Vector Vector6(6); 
        Vector6 << 1e-6 * score, 1e-6 * score, 1e-6 * score, 1e-5 * score, 1e-5 * score, 1e-5 * score; // 构成噪声协方差

        int from_id_i = genGlobalNodeIdx(priorSession_->index_, prior_id); // 源 ID 计算
        gtsam::Pose3 poseFrom_i = isamCurrentEstimate_.at<gtsam::Pose3>(from_id_i); // 获取源位姿

        int to_id = genGlobalNodeIdx(session_id, key_); // 目标 ID 计算
        gtsam::Pose3 poseRel_i = poseFrom_i.between(poseTo); // 计算目标位姿相对源位姿

        gtSAMgraph_.add(BetweenFactor<Pose3>(from_id_i, to_id, poseRel_i, matchNoise_)); // 将约束添加到图
    }
}

void updateSessionPoses() {
    // 此方法目前为空，会根据需要更新会话中的位姿
}

float getICPFitnessScore(const CloudPtr& cloud1_, const CloudPtr& cloud2_) {
    pcl::registration::CorrespondenceEstimation<PointType, PointType> est; // 新建配准相似度估计对象
    est.setInputSource(cloud1_); // 设置源点云
    est.setInputTarget(cloud2_); // 设置目标点云

    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences); // 没有共享的指针创建空间，用于保存所有配对
    est.determineCorrespondences(*correspondences); // 确定并存放配对关系

    float fitness_score = 0.0; // 初始得分设为0
    for (const auto& corr : *correspondences) { // 遍历所有配对
        fitness_score += corr.distance; // 累加每对的距离来评它们的适应度
    }

    fitness_score /= correspondences->size(); // 平均化得分

    return fitness_score; // 返回最终适应度分数
}
