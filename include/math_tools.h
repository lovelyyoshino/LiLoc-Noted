#ifndef SO3_MATH_H
#define SO3_MATH_H

#include <math.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#define SKEW_SYM_MATRX(v) 0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0

// hat
template<typename T>
// 将输入的三维向量转换为反对称矩阵（也称为“帽”操作）
Eigen::Matrix<T, 3, 3> skew_sym_mat(const Eigen::Matrix<T, 3, 1> &v)
{
    Eigen::Matrix<T, 3, 3> skew_sym_mat;
    // 构造反对称矩阵
    skew_sym_mat<<0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0;
    return skew_sym_mat;
}

template<typename T>
// 根据给定的旋转角度生成旋转矩阵，使用罗德里格斯公式，得到的是李群的指数映射
Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &&ang)
{
    T ang_norm = ang.norm(); // 计算角度的范数
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity(); // 单位矩阵
    if (ang_norm > 0.0000001) // 如果角度不接近零
    {
        Eigen::Matrix<T, 3, 1> r_axis = ang / ang_norm; // 归一化旋转轴
        Eigen::Matrix<T, 3, 3> K;
        K << SKEW_SYM_MATRX(r_axis); // 计算反对称矩阵K
        /// 罗德里格斯变换
        return Eye3 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K; // 返回旋转矩阵
    }
    else
    {
        return Eye3; // 当角度接近零时返回单位矩阵
    }
}

template<typename T, typename Ts>
// 使用角速度和时间增量生成旋转矩阵，使用罗德里格斯公式，得到的是李群的指数映射
Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &ang_vel, const Ts &dt)
{
    T ang_vel_norm = ang_vel.norm(); // 计算角速度的范数
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity(); // 单位矩阵

    if (ang_vel_norm > 0.0000001) // 如果角速度不接近零
    {
        Eigen::Matrix<T, 3, 1> r_axis = ang_vel / ang_vel_norm; // 归一化旋转轴
        Eigen::Matrix<T, 3, 3> K;

        K << SKEW_SYM_MATRX(r_axis); // 计算反对称矩阵K

        T r_ang = ang_vel_norm * dt; // 计算实际旋转角度

        /// 罗德里格斯变换
        return Eye3 + std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K; // 返回旋转矩阵
    }
    else
    {
        return Eye3; // 当角速度接近零时返回单位矩阵
    }
}

template<typename T>
// 根据三个分量生成旋转矩阵，使用罗德里格斯公式，得到的是李群的指数映射
Eigen::Matrix<T, 3, 3> Exp(const T &v1, const T &v2, const T &v3)
{
    T &&norm = sqrt(v1 * v1 + v2 * v2 + v3 * v3); // 计算旋转向量的范数
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity(); // 单位矩阵
    if (norm > 0.00001) // 如果范数不接近零
    {
        T r_ang[3] = {v1 / norm, v2 / norm, v3 / norm}; // 归一化旋转向量
        Eigen::Matrix<T, 3, 3> K;
        K << SKEW_SYM_MATRX(r_ang); // 计算反对称矩阵K

        /// 罗德里格斯变换
        return Eye3 + std::sin(norm) * K + (1.0 - std::cos(norm)) * K * K; // 返回旋转矩阵
    }
    else
    {
        return Eye3; // 当旋转向量接近零时返回单位矩阵
    }
}

/* 
 * @brief 计算旋转矩阵的对数映射，从李群到李代数的映射
 */
template<typename T>
Eigen::Matrix<T,3,1> Log(const Eigen::Matrix<T, 3, 3> &R)
{
    T theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1)); // 计算旋转角度
    Eigen::Matrix<T,3,1> K(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1)); // 计算旋转向量
    return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K); // 返回旋转向量
}

// 将旋转矩阵转换为欧拉角
template<typename T>
Eigen::Matrix<T, 3, 1> RotMtoEuler(const Eigen::Matrix<T, 3, 3> &rot)
{
    T sy = sqrt(rot(0,0)*rot(0,0) + rot(1,0)*rot(1,0)); // 计算sy
    bool singular = sy < 1e-6; // 判断是否奇异
    T x, y, z;
    if(!singular) // 非奇异情况
    {
        x = atan2(rot(2, 1), rot(2, 2)); // 计算x
        y = atan2(-rot(2, 0), sy);   // 计算y
        z = atan2(rot(1, 0), rot(0, 0));  // 计算z
    }
    else // 奇异情况
    {    
        x = atan2(-rot(1, 2), rot(1, 1));    // 计算x
        y = atan2(-rot(2, 0), sy);    // 计算y
        z = 0; // z设为0
    }
    Eigen::Matrix<T, 3, 1> ang(x, y, z); // 创建欧拉角向量
    return ang; // 返回欧拉角
}

/*
 * @brief 将给定四元数标准化为单位四元数。
 */
inline void quaternionNormalize(Eigen::Vector4d& q) {
  double norm = q.norm(); // 计算四元数的范数
  q = q / norm; // 标准化四元数
  return;
}

/*
 * @brief 执行四元数乘法 q1 * q2。
 *  
 *    q1 和 q2 的格式为 [x,y,z,w]
 */
inline Eigen::Vector4d quaternionMultiplication(
    const Eigen::Vector4d& q1,
    const Eigen::Vector4d& q2) {
  Eigen::Matrix4d L;

  // QXC: 哈密顿积
  L(0, 0) =  q1(3); L(0, 1) = -q1(2); L(0, 2) =  q1(1); L(0, 3) =  q1(0);
  L(1, 0) =  q1(2); L(1, 1) =  q1(3); L(1, 2) = -q1(0); L(1, 3) =  q1(1);
  L(2, 0) = -q1(1); L(2, 1) =  q1(0); L(2, 2) =  q1(3); L(2, 3) =  q1(2);
  L(3, 0) = -q1(0); L(3, 1) = -q1(1); L(3, 2) = -q1(2); L(3, 3) =  q1(3);

  Eigen::Vector4d q = L * q2; // 计算四元数乘法结果
  quaternionNormalize(q); // 标准化结果
  return q; // 返回结果
}

/*
 * @brief 将四元数的向量部分转换为完整的四元数。
 * @note 此函数用于将通常是3x1向量的增量四元数转换为完整的四元数。
 *       有关更多详细信息，请参见 "Indirect Kalman Filter for 3D Attitude Estimation:
 *       A Tutorial for quaternion Algebra" 中的第3.2节 "Kalman Filter Update"。
 */
inline Eigen::Vector4d smallAngleQuaternion(
    const Eigen::Vector3d& dtheta) {

  Eigen::Vector3d dq = dtheta / 2.0; // 计算增量四元数的一半
  Eigen::Vector4d q;
  double dq_square_norm = dq.squaredNorm(); // 计算平方范数

  if (dq_square_norm <= 1) {
    q.head<3>() = dq; // 设置四元数的前三个分量
    q(3) = std::sqrt(1-dq_square_norm); // 设置标量部分
  } else {
    q.head<3>() = dq; // 设置四元数的前三个分量
    q(3) = 1; // 标量部分设为1
    q = q / std::sqrt(1+dq_square_norm); // 标准化四元数
  }

  return q; // 返回四元数
}

/*
 * @brief 将四元数的向量部分转换为完整的四元数。
 * @note 此函数用于将通常是3x1向量的增量四元数转换为完整的四元数。
 *       有关更多详细信息，请参见 "Indirect Kalman Filter for 3D Attitude Estimation:
 *       A Tutorial for quaternion Algebra" 中的第3.2节 "Kalman Filter Update"。
 */
inline Eigen::Quaterniond getSmallAngleQuaternion(
    const Eigen::Vector3d& dtheta) {

  Eigen::Vector3d dq = dtheta / 2.0; // 计算增量四元数的一半
  Eigen::Quaterniond q;
  double dq_square_norm = dq.squaredNorm(); // 计算平方范数

  if (dq_square_norm <= 1) {
    q.x() = dq(0); // 设置四元数的x分量
    q.y() = dq(1); // 设置四元数的y分量
    q.z() = dq(2); // 设置四元数的z分量
    q.w() = std::sqrt(1-dq_square_norm); // 设置标量部分
  } else {
    q.x() = dq(0); // 设置四元数的x分量
    q.y() = dq(1); // 设置四元数的y分量
    q.z() = dq(2); // 设置四元数的z分量
    q.w() = 1; // 标量部分设为1
    q.normalize(); // 标准化四元数
  }

  return q; // 返回四元数
}

/*
 * @brief 将四元数转换为相应的旋转矩阵
 * @note 注意所用的约定。该函数遵循 "Indirect Kalman Filter for 3D Attitude Estimation:
 *       A Tutorial for Quaternion Algebra" 中的转换，方程(78)。
 *
 *       输入四元数应为形式
 *         [q1, q2, q3, q4(标量)]^T
 */
template <typename T>
inline Eigen::Matrix<T, 3, 3> quaternionToRotation(
    const Eigen::Matrix<T, 4, 1>& q) {
  // QXC: 哈密顿积
  const T& qw = q(3);
  const T& qx = q(0);
  const T& qy = q(1);
  const T& qz = q(2);
  Eigen::Matrix<T, 3, 3> R;
  // 根据四元数计算旋转矩阵
  R(0, 0) = 1-2*(qy*qy+qz*qz);  R(0, 1) =   2*(qx*qy-qw*qz);  R(0, 2) =   2*(qx*qz+qw*qy);
  R(1, 0) =   2*(qx*qy+qw*qz);  R(1, 1) = 1-2*(qx*qx+qz*qz);  R(1, 2) =   2*(qy*qz-qw*qx);
  R(2, 0) =   2*(qx*qz-qw*qy);  R(2, 1) =   2*(qy*qz+qw*qx);  R(2, 2) = 1-2*(qx*qx+qy*qy);

  return R; // 返回旋转矩阵
}

template <typename T>
// 将欧拉角转换为旋转矩阵
inline Eigen::Matrix<T, 3, 3> eulerToRotation(const T& roll, const T& pitch, const T& yaw) {
    Eigen::AngleAxis<T> rollAngle(roll, Eigen::Matrix<T, 3, 1>::UnitX()); // 绕X轴旋转
    Eigen::AngleAxis<T> pitchAngle(pitch, Eigen::Matrix<T, 3, 1>::UnitY()); // 绕Y轴旋转
    Eigen::AngleAxis<T> yawAngle(yaw, Eigen::Matrix<T, 3, 1>::UnitZ()); // 绕Z轴旋转

    // 按照顺序组合旋转矩阵
    Eigen::Matrix<T, 3, 3> rotation = yawAngle.toRotationMatrix() * pitchAngle.toRotationMatrix() * rollAngle.toRotationMatrix();

    return rotation; // 返回最终的旋转矩阵
}

/*
 * @brief 将旋转矩阵转换为四元数。
 * @note 注意所用的约定。该函数遵循 "Indirect Kalman Filter for 3D Attitude Estimation:
 *       A Tutorial for Quaternion Algebra" 中的转换，方程(78)。
 *
 *       输入四元数应为形式
 *         [q1, q2, q3, q4(标量)]^T
 */
inline Eigen::Vector4d rotationToQuaternion(
    const Eigen::Matrix3d& R) {
  Eigen::Vector4d score;
  score(0) = R(0, 0);
  score(1) = R(1, 1);
  score(2) = R(2, 2);
  score(3) = R.trace(); // 计算旋转矩阵的迹

  int max_row = 0, max_col = 0;
  score.maxCoeff(&max_row, &max_col); // 找到最大元素的位置

  Eigen::Vector4d q = Eigen::Vector4d::Zero();

  // QXC: 哈密顿积
  if (max_row == 0) {
    q(0) = std::sqrt(1+2*R(0, 0)-R.trace()) / 2.0; // 计算q的x分量
    q(1) = (R(0, 1)+R(1, 0)) / (4*q(0)); // 计算q的y分量
    q(2) = (R(0, 2)+R(2, 0)) / (4*q(0)); // 计算q的z分量
    q(3) = (R(2, 1)-R(1, 2)) / (4*q(0)); // 计算q的w分量
  } else if (max_row == 1) {
    q(1) = std::sqrt(1+2*R(1, 1)-R.trace()) / 2.0; // 计算q的y分量
    q(0) = (R(0, 1)+R(1, 0)) / (4*q(1)); // 计算q的x分量
    q(2) = (R(1, 2)+R(2, 1)) / (4*q(1)); // 计算q的z分量
    q(3) = (R(0, 2)-R(2, 0)) / (4*q(1)); // 计算q的w分量
  } else if (max_row == 2) {
    q(2) = std::sqrt(1+2*R(2, 2)-R.trace()) / 2.0; // 计算q的z分量
    q(0) = (R(0, 2)+R(2, 0)) / (4*q(2)); // 计算q的x分量
    q(1) = (R(1, 2)+R(2, 1)) / (4*q(2)); // 计算q的y分量
    q(3) = (R(1, 0)-R(0, 1)) / (4*q(2)); // 计算q的w分量
  } else {
    q(3) = std::sqrt(1+R.trace()) / 2.0; // 计算q的w分量
    q(0) = (R(2, 1)-R(1, 2)) / (4*q(3)); // 计算q的x分量
    q(1) = (R(0, 2)-R(2, 0)) / (4*q(3)); // 计算q的y分量
    q(2) = (R(1, 0)-R(0, 1)) / (4*q(3)); // 计算q的z分量
  }

  if (q(3) < 0) q = -q; // 确保四元数的标量部分非负
  quaternionNormalize(q); // 标准化四元数
  return q; // 返回四元数
}

// 符号函数
template <typename T>
T sgnFunc(T val)
{
    return (T(0) < val) - (val < T(0)); // 返回值的符号
}

template <typename Derived>
// 计算给定向量的反对称矩阵
static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
{
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1),
            q(2), typename Derived::Scalar(0), -q(0),
            -q(1), q(0), typename Derived::Scalar(0);
    return ans; // 返回反对称矩阵
}

template <typename Derived>
// 计算左侧四元数矩阵，左四元数矩阵是四元数乘法的一种表示形式，矩阵包含了四元数的旋转部分和一个额外的维度，适用于更复杂的变换，例如在齐次坐标系中使用
static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
{
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;//将其初始化为零矩阵
    ans(0, 0) = q.w(), ans.template block<1, 3>(0, 1) = -q.vec().transpose();//将第一行的其余元素设置为四元数的虚部的相反数
    ans.template block<3, 1>(1, 0) = q.vec(), ans.template block<3, 3>(1, 1) = q.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + skewSymmetric(q.vec());//将矩阵的第一列的其余元素设置为四元数的虚部
    return ans; // 返回左侧四元数矩阵
}

template <typename Derived>
// 计算右侧四元数矩阵，矩阵包含了四元数的旋转部分和一个额外的维度，适用于更复杂的变换，例如在齐次坐标系中使用
/*
    // 定义一个四元数
    Eigen::Quaternionf q(0.7071f, 0.0f, 0.7071f, 0.0f); // 代表绕 Z 轴旋转 90 度
    
    // 计算右侧四元数矩阵
    Eigen::Matrix4f Q = Qright(q);
    
    // 输出 Qright 矩阵
    std::cout << "Qright Matrix:" << std::endl;
    std::cout << Q << std::endl;

    // 旋转一个点
    Eigen::Vector3f point(1.0f, 0.0f, 0.0f); // 要旋转的点
    Eigen::Vector4f point_h(point[0], point[1], point[2], 1.0f); // 齐次坐标

    // 计算旋转后的点
    Eigen::Vector4f rotated_point = Q * point_h;

    // 输出旋转后的点
    std::cout << "Rotated Point:" << std::endl;
    std::cout << rotated_point.head<3>() << std::endl; // 只输出前三个分量

*/
static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p)
{
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = p.w(), ans.template block<1, 3>(0, 1) = -p.vec().transpose();
    ans.template block<3, 1>(1, 0) = p.vec(), ans.template block<3, 3>(1, 1) = p.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() - skewSymmetric(p.vec());
    return ans; // 返回右侧四元数矩阵
}

// 从四元数转换为旋转向量
template <typename T>
inline Eigen::Matrix<T, 3, 1> quaternionToRotationVector(const Eigen::Quaternion<T> &qua)
{
    Eigen::Matrix<T, 3, 3> mat = qua.toRotationMatrix(); // 获取旋转矩阵
    Eigen::Matrix<T, 3, 1> rotation_vec; // 初始化旋转向量
    Eigen::AngleAxis<T> angle_axis; // 定义角轴对象
    angle_axis.fromRotationMatrix(mat); // 从旋转矩阵获取角轴
    rotation_vec = angle_axis.angle() * angle_axis.axis(); // 计算旋转向量
    return rotation_vec; // 返回旋转向量
}


// 右雅可比矩阵
template <typename T>
inline Eigen::Matrix3d Jright(const Eigen::Quaternion<T> &qua)
{
    Eigen::Matrix<T, 3, 3> mat;
    Eigen::Matrix<T, 3, 1> rotation_vec = quaternionToRotationVector(qua); // 获取旋转向量
    double theta_norm = rotation_vec.norm(); // 计算旋转向量的范数
    mat = Eigen::Matrix<T, 3, 3>::Identity()
            - (1 - cos(theta_norm)) / (theta_norm * theta_norm + 1e-10) * hat(rotation_vec) // 计算雅可比矩阵
            + (theta_norm - sin(theta_norm)) / (theta_norm * theta_norm * theta_norm + 1e-10) * hat(rotation_vec) * hat(rotation_vec);
    return mat; // 返回雅可比矩阵
}

// 计算关于四元数的雅可比矩阵，输入参数包括一个四元数 qua 和一个三维向量 vec，表示四元数和三维向量之间的
/*
    // 假设你有一个四元数和一个向量
    Eigen::Quaterniond qua(0.7071, 0.0, 0.7071, 0.0); // 代表绕 Z 轴旋转 90 度
    Eigen::Vector3d vec(1.0, 0.0, 0.0); // 一个向量

    // 计算雅可比矩阵
    Eigen::Matrix<double, 3, 4> jacobian = quaternionJacobian(qua, vec);

    // 假设你有一个微小的四元数变化量 dq
    Eigen::Quaterniond dq(1.0, 0.01, 0.0, 0.0); // 一个微小的四元数变化量

    // 将 dq 转换为向量形式
    Eigen::Vector4d dq_vec(dq.w(), dq.x(), dq.y(), dq.z());

    // 使用雅可比矩阵计算向量的变化量
    Eigen::Vector3d vec_change = jacobian * dq_vec;

    // 输出结果
    std::cout << "Original vector: " << vec.transpose() << std::endl;
    std::cout << "Vector change: " << vec_change.transpose() << std::endl;
    std::cout << "New vector: " << (vec + vec_change).transpose() << std::endl;
*/
template <typename T>
inline Eigen::Matrix<T, 3, 4> quaternionJacobian(const Eigen::Quaternion<T> &qua, const Eigen::Matrix<T, 3, 1> &vec)
{
    Eigen::Matrix<T, 3, 4> mat;
    Eigen::Matrix<T, 3, 1> quaternion_imaginary(qua.x(), qua.y(), qua.z());

    mat.template block<3, 1>(0, 0) = qua.w() * vec + quaternion_imaginary.cross(vec); // 计算第一列
    mat.template block<3, 3>(0, 1) = quaternion_imaginary.dot(vec) * Eigen::Matrix<T, 3, 3>::Identity()
            + quaternion_imaginary * vec.transpose()
            - vec * quaternion_imaginary.transpose()
            - qua.w() * hat(vec); // 计算后面三列，这三列的计算是通过将四元数虚部与输入向量的点积乘以单位矩阵，然后加上四元数虚部与输入向量的外积，再减去输入向量与四元数虚部的外积，最后减去四元数实部与输入向量的反对称矩阵得到的。
    return T(2) * mat; // 返回雅可比矩阵，四元数的导数计算中，通常需要将结果乘以 2
}

// 计算关于逆四元数的雅可比矩阵
template <typename T>
inline Eigen::Matrix<T, 3, 4> quaternionInvJacobian(const Eigen::Quaternion<T> &qua, const Eigen::Matrix<T, 3, 1> &vec)
{
    Eigen::Matrix<T, 3, 4> mat;
    Eigen::Matrix<T, 3, 1> quaternion_imaginary(qua.x(), qua.y(), qua.z());

    mat.template block<3, 1>(0, 0) = qua.w() * vec + vec.cross(quaternion_imaginary); // 计算第一列
    mat.template block<3, 3>(0, 1) = quaternion_imaginary.dot(vec) * Eigen::Matrix<T, 3, 3>::Identity()
            + quaternion_imaginary * vec.transpose()
            - vec * quaternion_imaginary.transpose()
            + qua.w() * hat(vec); // 计算后面三列
    return T(2) * mat; // 返回雅可比矩阵
}

// 计算从旋转向量到四元数的雅可比矩阵
template <typename T>
inline Eigen::Matrix<T, 3, 4> JacobianV2Q(const Eigen::Quaternion<T> &qua)
{
    Eigen::Matrix<T, 3, 4> mat;

    T c = 1 / (1 - qua.w() * qua.w()); // 计算常数c
    T d = acos(qua.w()) / sqrt(1 - qua.w() * qua.w()); // 计算常数d

    mat.template block<3, 1>(0, 0) = Eigen::Matrix<T, 3, 1>(c * qua.x() * (d * qua.x() - 1),
                                                            c * qua.y() * (d * qua.x() - 1),
                                                            c * qua.z() * (d * qua.x() - 1)); // 计算第一列
    mat.template block<3, 3>(0, 1) = d * Eigen::Matrix<T, 3, 4>::Identity(); // 计算后面三列
    return T(2) * mat; // 返回雅可比矩阵
}

// 从旋转向量获取四元数
template <typename Derived>
Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
{
    typedef typename Derived::Scalar Scalar_t;

    Eigen::Quaternion<Scalar_t> dq;
    Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta; // 计算旋转向量的一半
    half_theta /= static_cast<Scalar_t>(2.0);
    dq.w() = static_cast<Scalar_t>(1.0); // 设置四元数的w分量
    dq.x() = half_theta.x(); // 设置四元数的x分量
    dq.y() = half_theta.y(); // 设置四元数的y分量
    dq.z() = half_theta.z(); // 设置四元数的z分量
    return dq; // 返回四元数
}

// 计算四元数的左旋转矩阵
template<typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 4, 4> LeftQuatMatrix(const Eigen::QuaternionBase<Derived> &q) {
    Eigen::Matrix<typename Derived::Scalar, 4, 4> m;
    Eigen::Matrix<typename Derived::Scalar, 3, 1> vq = q.vec(); // 获取四元数的向量部分
    typename Derived::Scalar q4 = q.w(); // 获取四元数的标量部分
    m.block(0, 0, 3, 3) << q4 * Eigen::Matrix3d::Identity() + skewSymmetric(vq); // 填充左侧四元数矩阵
    m.block(3, 0, 1, 3) << -vq.transpose(); // 填充最后一行
    m.block(0, 3, 3, 1) << vq; // 填充最后一列
    m(3, 3) = q4; // 设置最后一个元素
    return m; // 返回左侧四元数矩阵
}

// 计算四元数的右旋转矩阵
template<typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 4, 4> RightQuatMatrix(const Eigen::QuaternionBase<Derived> &p) {
    Eigen::Matrix<typename Derived::Scalar, 4, 4> m;
    Eigen::Matrix<typename Derived::Scalar, 3, 1> vp = p.vec(); // 获取四元数的向量部分
    typename Derived::Scalar p4 = p.w(); // 获取四元数的标量部分
    m.block(0, 0, 3, 3) << p4 * Eigen::Matrix3d::Identity() - skewSymmetric(vp); // 填充右侧四元数矩阵
    m.block(3, 0, 1, 3) << -vp.transpose(); // 填充最后一行
    m.block(0, 3, 3, 1) << vp; // 填充最后一列
    m(3, 3) = p4; // 设置最后一个元素
    return m; // 返回右侧四元数矩阵
}

// 统一四元数，使其标量部分为正
template <typename T>
Eigen::Quaternion<T> unifyQuaternion(const Eigen::Quaternion<T> &q)
{
    if(q.w() >= 0) return q; // 如果标量部分非负，直接返回
    else {
        Eigen::Quaternion<T> resultQ(-q.w(), -q.x(), -q.y(), -q.z()); // 否则取反
        return resultQ; // 返回结果
    }
}

#endif
