#pragma once

#ifndef _DATA_LOADER_
#define _DATA_LOADER_

#include "../utility.h"

#include <experimental/filesystem> // file gcc>=8
#include <experimental/optional>

namespace fs = std::experimental::filesystem;

namespace dataManager {

// 边结构体
struct Edge {
    int from_idx; // 边的起始节点索引
    int to_idx;   // 边的结束节点索引
    gtsam::Pose3 relative; // 相对位姿
};

// 节点结构体
struct Node {
    int idx; // 节点索引
    gtsam::Pose3 initial; // 初始位姿
};

// 位姿结构体
struct Pose
{
    Eigen::Vector3d t; // 平移向量
    Eigen::Matrix3d R; // 旋转矩阵
};

// g2o格式信息结构体
struct G2oLineInfo {
    std::string type; // 类型（边或节点）

    int prev_idx = -1; // 前一个节点索引，若为顶点则为空
    int curr_idx; // 当前节点索引

    std::vector<double> trans; // 平移部分
    std::vector<double> quat;  // 四元数表示的旋转部分

    inline static const std::string kVertexTypeName = "VERTEX_SE3:QUAT"; // 顶点类型名称
    inline static const std::string kEdgeTypeName = "EDGE_SE3:QUAT";     // 边类型名称
}; 

using SessionNodes = std::multimap<int, Node>; // 从索引到Node的多重映射
using SessionEdges = std::multimap<int, Edge>; // 从索引到Edge的多重映射

// 判断两个字符串是否相同
bool isTwoStringSame(std::string _str1, std::string _str2) {
	return !(_str1.compare(_str2)); // 如果比较结果为0，则返回true
}

// 读取g2o文件行并解析
// 示例：VERTEX_SE3:QUAT 99 -61.332581 -9.253125 0.131973 -0.004256 -0.005810 -0.625732 0.780005
G2oLineInfo splitG2oFileLine(std::string _str_line) {

    std::stringstream ss(_str_line); // 创建字符串流以分割字符串

	std::vector<std::string> parsed_elms ; // 存储解析后的元素
    std::string elm;
	char delimiter = ' '; // 定义分隔符
    while (getline(ss, elm, delimiter)) { // 按照分隔符逐个获取元素
        parsed_elms.push_back(elm); // 将元素添加到解析列表中
    }

	G2oLineInfo parsed; // 创建G2oLineInfo对象用于存储解析结果
    // 确定是边还是节点
	if (isTwoStringSame(parsed_elms.at(0), G2oLineInfo::kVertexTypeName))
	{
		parsed.type = parsed_elms.at(0);// 顶点类型
		parsed.curr_idx = std::stoi(parsed_elms.at(1));// 当前索引
		parsed.trans.push_back(std::stod(parsed_elms.at(2)));
		parsed.trans.push_back(std::stod(parsed_elms.at(3)));
		parsed.trans.push_back(std::stod(parsed_elms.at(4)));
		parsed.quat.push_back(std::stod(parsed_elms.at(5)));
		parsed.quat.push_back(std::stod(parsed_elms.at(6)));
		parsed.quat.push_back(std::stod(parsed_elms.at(7)));
		parsed.quat.push_back(std::stod(parsed_elms.at(8)));
	}
	if (isTwoStringSame(parsed_elms.at(0), G2oLineInfo::kEdgeTypeName))
	{
		parsed.type = parsed_elms.at(0);// 边类型
		parsed.prev_idx = std::stoi(parsed_elms.at(1));// 前一个索引
		parsed.curr_idx = std::stoi(parsed_elms.at(2));// 当前索引
		parsed.trans.push_back(std::stod(parsed_elms.at(3)));
		parsed.trans.push_back(std::stod(parsed_elms.at(4)));
		parsed.trans.push_back(std::stod(parsed_elms.at(5)));
		parsed.quat.push_back(std::stod(parsed_elms.at(6)));
		parsed.quat.push_back(std::stod(parsed_elms.at(7)));
		parsed.quat.push_back(std::stod(parsed_elms.at(8)));
		parsed.quat.push_back(std::stod(parsed_elms.at(9)));
	}

	return parsed; // 返回解析结果
}

// 显示进度条
void showProgressBar(int progress, int total) {
    int barWidth = 70; 
    float progressRatio = static_cast<float>(progress) / total; // 计算进度比例

    std::cout << "[";
    int pos = barWidth * progressRatio; // 计算当前进度在进度条中的位置
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "="; // 已完成部分
        else if (i == pos) std::cout << ">"; // 当前进度指示器
        else std::cout << " "; // 未完成部分
    }
    std::cout << "] " << int(progressRatio * 100.0) << " %\r"; // 输出百分比
    std::cout.flush(); // 刷新输出
}

// 根据文件名排序
bool fileNameSort(std::string name1_, std::string name2_){
    std::string::size_type iPos1 = name1_.find_last_of('/') + 1; // 找到最后一个斜杠的位置
	std::string filename1 = name1_.substr(iPos1, name1_.length() - iPos1); // 获取文件名
	std::string name1 = filename1.substr(0, filename1.rfind(".")); // 去掉扩展名

    std::string::size_type iPos2 = name2_.find_last_of('/') + 1;
    std::string filename2 = name2_.substr(iPos2, name2_.length() - iPos2);
	std::string name2 = filename2.substr(0, filename2.rfind(".")); // 去掉扩展名

    return std::stoi(name1) < std::stoi(name2); // 比较文件名数字大小
}

// 会话类，继承自参数服务器
class Session : public ParamServer {
public:
    int index_; // 会话索引

    std::string name_; // 会话名称
    std::string session_dir_path_; // 会话目录路径

    bool is_base_session_; // 是否为基础会话

    SessionNodes nodes_; // 节点集合
    SessionEdges edges_; // 边集合

    int anchor_node_idx_; // 锚节点索引

    TrajectoryPtr KeyPoses6D_;  // 解析的关键位姿
    TrajectoryPtr originPoses6D_; // 原始位姿

    CloudPtr globalMap_; // 全局地图

    std::vector<CloudPtr> keyCloudVec_; // 关键点云集合

    int prior_size; // 先前大小

    std::vector<TrajectoryPtr> subMapVertexCloudVec_; // 子图顶点点云集合
    std::vector<CloudPtr> subMapCloudVec_; // 子图点云集合
    TrajectoryPtr SubMapCenteriod_; // 子图中心轨迹

    CloudPtr usingSubMap_; // 使用的子图
    TrajectoryPtr usingVertexes_; // 使用的顶点

    pcl::KdTreeFLANN<PointTypePose>::Ptr kdtreeSearchVertex_; // 用于搜索顶点的KD树
    pcl::KdTreeFLANN<PointTypePose>::Ptr kdtreeSearchSubMap_; // 用于搜索子图的KD树

    int submap_id = -1; // 当前子图ID
    int submap_size = 10; // 子图大小
    int search_gap = submap_size; // 搜索间隔

    bool margFlag = false; // 边缘标志

    std::vector<int> localGoup_; // 本地组

    pcl::VoxelGrid<PointType> downSizeFilterSurf; // 点云下采样过滤器

public:
    ~Session() { } // 析构函数
    Session() { } // 默认构造函数

    // 带参数的构造函数
    Session(int _idx, std::string _name, std::string _session_dir_path, bool _is_base_session)
           : index_(_idx), name_(_name), session_dir_path_(_session_dir_path), is_base_session_(_is_base_session){

        allocateMemory(); // 分配内存

        loadSessionGraph(); // 加载会话图

        loadGlobalMap(); // 加载全局地图

        loadKeyCloud(); // 加载关键点云

        prior_size = KeyPoses6D_->size(); // 设置先前大小

        generateSubMaps(); // 生成子图

        ROS_INFO_STREAM("\033[1;32m Session " << index_ << " (" << name_ << ") is loaded successfully \033[0m");
    }

    // 内存分配函数
    void allocateMemory() {
        KeyPoses6D_.reset(new Trajectory()); // 初始化关键位姿
        originPoses6D_.reset(new Trajectory()); // 初始化原始位姿
        globalMap_.reset(new Cloud()); // 初始化全局地图

        usingSubMap_.reset(new Cloud()); // 初始化使用的子图
        usingVertexes_.reset(new Trajectory()); // 初始化使用的顶点

        SubMapCenteriod_.reset(new Trajectory()); // 初始化子图中心轨迹

        kdtreeSearchSubMap_.reset(new pcl::KdTreeFLANN<PointTypePose>()); // 初始化KD树
        kdtreeSearchVertex_.reset(new pcl::KdTreeFLANN<PointTypePose>()); // 初始化KD树
    }

    // 加载会话图，当中是以g2o格式存储的
    void loadSessionGraph() {
        std::string posefile_path = session_dir_path_ + "/singlesession_posegraph.g2o"; // 定义pose文件路径

        std::ifstream posefile_handle (posefile_path); // 打开pose文件
        std::string strOneLine;

        while (getline(posefile_handle, strOneLine)) { // 逐行读取文件
            G2oLineInfo line_info = splitG2oFileLine(strOneLine); // 解析每一行

            // 保存变量（节点）
            if (isTwoStringSame(line_info.type, G2oLineInfo::kVertexTypeName)) {
                Node this_node { line_info.curr_idx, gtsam::Pose3( 
                    gtsam::Rot3(gtsam::Quaternion(line_info.quat[3], line_info.quat[0], line_info.quat[1], line_info.quat[2])), // xyzw 转 wxyz
                    gtsam::Point3(line_info.trans[0], line_info.trans[1], line_info.trans[2])) }; 
                nodes_.insert(std::pair<int, Node>(line_info.curr_idx, this_node)); // 插入节点
            }
 
            // 保存边
            if(isTwoStringSame(line_info.type, G2oLineInfo::kEdgeTypeName)) {
                Edge this_edge { line_info.prev_idx, line_info.curr_idx, gtsam::Pose3( 
                    gtsam::Rot3(gtsam::Quaternion(line_info.quat[3], line_info.quat[0], line_info.quat[1], line_info.quat[2])), // xyzw 转 wxyz
                    gtsam::Point3(line_info.trans[0], line_info.trans[1], line_info.trans[2])) }; 
                edges_.insert(std::pair<int, Edge>(line_info.prev_idx, this_edge)); // 插入边
            }
        }

        initKeyPoses(); // 初始化关键位姿

        ROS_INFO_STREAM("\033[1;32m Graph loaded: " << posefile_path << " - num nodes: " << nodes_.size() << "\033[0m"); // 输出加载信息
    }

    // 初始化关键位姿
    void initKeyPoses() {
        for (auto & _node_info: nodes_) {
            PointTypePose thisPose6D;

            int node_idx = _node_info.first; // 获取节点索引
            Node node = _node_info.second; // 获取节点
            gtsam::Pose3 pose = node.initial; // 获取初始位姿

            thisPose6D.x = pose.translation().x(); // 设置平移部分
            thisPose6D.y = pose.translation().y();
            thisPose6D.z = pose.translation().z();
            thisPose6D.intensity = node_idx; // TODO: 设置强度
            thisPose6D.roll  = pose.rotation().roll(); // 设置滚转角
            thisPose6D.pitch = pose.rotation().pitch(); // 设置俯仰角
            thisPose6D.yaw   = pose.rotation().yaw(); // 设置偏航角
            thisPose6D.time = 0.0; // TODO: 不使用时间

            KeyPoses6D_->push_back(thisPose6D); // 添加到关键位姿中，从G2O文件中读取
        }

        // 添加原点位姿
        PointTypePose thisPose6D;
        thisPose6D.x = 0.0;
        thisPose6D.y = 0.0;
        thisPose6D.z = 0.0;
        thisPose6D.intensity = 0.0;
        thisPose6D.roll = 0.0;
        thisPose6D.pitch = 0.0;
        thisPose6D.yaw = 0.0;
        thisPose6D.time = 0.0;
        originPoses6D_->push_back(thisPose6D); // 添加到原始位姿中
    }

    // 加载全局地图
    void loadGlobalMap() {
        std::string mapfile_path = session_dir_path_ + "/globalMap.pcd";  
        pcl::io::loadPCDFile<PointType>(mapfile_path, *globalMap_); // 从文件加载点云数据
        ROS_INFO_STREAM("\033[1;32m Map loaded: " << mapfile_path << " - size: " << globalMap_->points.size() << "\033[0m"); // 输出加载信息
    }

    // 加载关键点云
    void loadKeyCloud() {
        std::string pcd_dir = session_dir_path_ + "/PCDs/"; // 定义点云目录

        std::vector<std::string> pcd_names; // 存储点云文件名
        for(auto& pcd : fs::directory_iterator(pcd_dir)) { // 遍历目录中的所有文件
            std::string pcd_filepath = pcd.path(); // 获取文件路径
            pcd_names.emplace_back(pcd_filepath); // 添加到文件名列表
        }

        std::sort(pcd_names.begin(), pcd_names.end(), fileNameSort); // 对文件名进行排序

        int pcd_num = pcd_names.size(); // 获取点云数量

        int pcd_count = 0; 
        for (auto const& pcd_name: pcd_names){ // 遍历每个点云文件
            pcl::PointCloud<PointType>::Ptr this_pcd(new pcl::PointCloud<PointType>());   
            pcl::io::loadPCDFile<PointType> (pcd_name, *this_pcd); // 加载点云文件

            showProgressBar(pcd_count, pcd_num); // 显示进度条
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 暂停1毫秒

            keyCloudVec_.push_back(this_pcd); // 添加到关键点云集合

            pcd_count++; // 增加计数
        }

        ROS_INFO_STREAM("\033[1;32m Key Cloud loaded: " << pcd_dir << " - num pcds: " << keyCloudVec_.size() << "\033[0m"); // 输出加载信息
    }

    // 生成子图
    void generateSubMaps() {
        CloudPtr submap(new Cloud()); // 创建新的子图
        float x = 0.0, y = 0.0, z = 0.0; // 坐标累加
        int count = 0; // 计数器

        TrajectoryPtr vertexCloud(new Trajectory()); // 创建顶点点云

        for (int i = 0; i < keyCloudVec_.size(); i++) { // 遍历关键点云，这里是将历史保存下来的G2O来生成子图
            count++;
            *submap += *transformPointCloud(keyCloudVec_[i], &KeyPoses6D_->points[i]); // 转换并累加点云，和G2O文件中的位姿对应
            x += KeyPoses6D_->points[i].x; // 累加坐标
            y += KeyPoses6D_->points[i].y;
            z += KeyPoses6D_->points[i].z;

            vertexCloud->push_back(KeyPoses6D_->points[i]); // 添加到顶点点云

            if (count % submap_size == 0 || i == keyCloudVec_.size() - 1) { // 每达到一定数量或最后一个时
                PointTypePose centeriod; // 创建子图中心点
                centeriod.x = x / (float)count; // 计算平均值
                centeriod.y = y / (float)count;
                centeriod.z = z / (float)count;
                SubMapCenteriod_->push_back(centeriod); // 添加到子图中心轨迹

                subMapVertexCloudVec_.push_back(vertexCloud); // 添加到子图顶点点云集合
                CloudPtr submap_copy(new Cloud()); // 创建子图副本

                downSizeFilterSurf.setLeafSize(0.2, 0.2, 0.2); // 设置下采样叶子大小
                downSizeFilterSurf.setInputCloud(submap); // 设置输入点云
                downSizeFilterSurf.filter(*submap_copy); // 执行下采样

                pcl::copyPointCloud(*submap, *submap_copy); // 拷贝点云
                subMapCloudVec_.push_back(submap_copy); // 添加到子图点云集合

                count = 0; // 重置计数器
                x = 0.0; y = 0.0; z = 0.0; // 重置坐标累加
                submap->clear(); // 清空子图
            }
        }

        ROS_INFO_STREAM("\033[1;32m Submap Generated - num: " << subMapCloudVec_.size() << " with pcd num: " << submap_size << "\033[0m"); // 输出生成信息
    }

    // 查找最近的子图和顶点
    void searchNearestSubMapAndVertex(const PointTypePose& pose, int& map_id) {
        std::vector<int> ids; // 存储找到的ID
        std::vector<float> dis; // 存储距离
        kdtreeSearchSubMap_->setInputCloud(SubMapCenteriod_); // 设置KD树输入
        kdtreeSearchSubMap_->nearestKSearchT(pose, 1, ids, dis); // 查找最近的子图

        map_id = ids[0]; // 获取子图ID
        if (submap_id != map_id) { // 如果子图ID发生变化
            submap_id = map_id; // 更新子图ID
            margFlag = true; // 设置边缘标志，因为在addNewPrior函数中完成了SubMapCenteriod_的生成，最近的肯定在最新生成的附近
        }
        
        usingSubMap_ = subMapCloudVec_[map_id]; // 使用找到的子图

        std::cout << usingSubMap_->size() << std::endl; // 输出子图大小

        ids.clear(); dis.clear(); // 清空ID和距离
        kdtreeSearchVertex_->setInputCloud(subMapVertexCloudVec_[map_id]); // 设置KD树输入，这里面存放的是G2O文件中的位姿合集
        kdtreeSearchVertex_->radiusSearch(pose, 5.0, ids, dis); // 半径搜索顶点，找到附近的关键点

        usingVertexes_->clear(); // 清空使用的顶点
        localGoup_.clear(); // 清空本地组
        int count = 0; // 计数器
        for (auto id : ids) { // 遍历找到的ID
            int curNew = KeyPoses6D_->size(); // 获取当前关键位姿数量

            // 检查条件，确保关键点是递增的，且当前点和之前的点的差值不超过10
            if (subMapVertexCloudVec_[map_id]->points[id].intensity > prior_size && 
                std::abs(curNew - subMapVertexCloudVec_[map_id]->points[id].intensity) <= 10) 
            {
                continue; // 跳过不符合条件的顶点
            }

            count++;

            if (count > 3) { // 限制最大数量
                break;
            }
            usingVertexes_->push_back(subMapVertexCloudVec_[map_id]->points[id]); // 添加到使用的顶点
            localGoup_.push_back(id); // 添加到本地组
        }
    }
};

}

#endif