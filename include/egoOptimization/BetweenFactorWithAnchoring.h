#pragma once

#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

namespace gtsam {

  /**
   * 一个由 "between(config[key1], config[key2])" 预测的测量类
   * @tparam VALUE 值类型
   * @addtogroup SLAM
   */
  template<class VALUE>
  class BetweenFactorWithAnchoring: public NoiseModelFactor4<VALUE, VALUE, VALUE, VALUE> {

    // 检查VALUE类型是否是可测试的李群
    BOOST_CONCEPT_ASSERT((IsTestable<VALUE>));
    BOOST_CONCEPT_ASSERT((IsLieGroup<VALUE>));

  public:

    typedef VALUE T;

  private:

    typedef BetweenFactorWithAnchoring<VALUE> This; // 当前类的别名
    typedef NoiseModelFactor4<VALUE, VALUE, VALUE, VALUE> Base; // 基类的别名

    VALUE measured_; /** 测量值 */

  public:

    // 智能指针的简写
    typedef typename boost::shared_ptr<BetweenFactorWithAnchoring> shared_ptr;

    /** 默认构造函数 - 仅用于序列化 */
    BetweenFactorWithAnchoring() {}

    /** 构造函数 */
    BetweenFactorWithAnchoring(
          // 机器人1和机器人2的键
          Key key1, Key key2, Key anchor_key1, Key anchor_key2,
          const VALUE& measured,
          const SharedNoiseModel& model = nullptr) :
      Base(model, key1, key2, anchor_key1, anchor_key2), measured_(measured) {
    }

    virtual ~BetweenFactorWithAnchoring() {}

    /// @return 此因子的深拷贝
    virtual gtsam::NonlinearFactor::shared_ptr clone() const {
      return boost::static_pointer_cast<gtsam::NonlinearFactor>(
          gtsam::NonlinearFactor::shared_ptr(new This(*this))); 
    }

    /** 实现 Testable 所需的函数 */

    /** 打印 */
    virtual void print(const std::string& s, const KeyFormatter& keyFormatter = DefaultKeyFormatter) const {
      std::cout << s << "BetweenFactorWithAnchoring("
          << keyFormatter(this->key1()) << ","
          << keyFormatter(this->key2()) << ")\n";
      traits<T>::Print(measured_, "  measured: ");
      this->noiseModel_->print("  noise model: ");
    }

    /** 等于比较 */
    virtual bool equals(const NonlinearFactor& expected, double tol=1e-9) const {
      const This *e =  dynamic_cast<const This*> (&expected);
      return e != nullptr && Base::equals(*e, tol) && traits<T>::Equals(this->measured_, e->measured_, tol);
    }

    /** 实现 Factor 所需的函数 */

    /** 计算误差向量 */
    Vector evaluateError(
        const T& p1, const T& p2, const T& anchor_p1, const T& anchor_p2,
        boost::optional<Matrix&> H1 = boost::none,
        boost::optional<Matrix&> H2 = boost::none,
        boost::optional<Matrix&> anchor_H1 = boost::none,
        boost::optional<Matrix&> anchor_H2 = boost::none
      ) const {

      // 锚节点 h(.) (参考: isam ver. line 233)
      T hx1 = traits<T>::Compose(anchor_p1, p1, anchor_H1, H1); // 更新雅可比，见相关文档
      T hx2 = traits<T>::Compose(anchor_p2, p2, anchor_H2, H2); 
      T hx = traits<T>::Between(hx1, hx2, H1, H2); 

      return traits<T>::Local(measured_, hx); // 返回局部误差
    }

    /** 返回测量值 */
    const VALUE& measured() const {
      return measured_;
    }

    /** 附加到此因子的变量数量 */
    std::size_t size() const {
      return 4; // 因子涉及四个变量
    }

  private:

    /** 序列化函数 */
    friend class boost::serialization::access;
    template<class ARCHIVE>
    void serialize(ARCHIVE & ar, const unsigned int /*version*/) {
      ar & boost::serialization::make_nvp("NoiseModelFactor4",
          boost::serialization::base_object<Base>(*this));
      ar & BOOST_SERIALIZATION_NVP(measured_);
    }

	//   // 对齐，参见 https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
	//   enum { NeedsToAlign = (sizeof(VALUE) % 16) == 0 };
  //   public:
  //     EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(NeedsToAlign)
  }; // \class BetweenFactorWithAnchoring


  // traits
  template<class VALUE>
  struct traits<BetweenFactorWithAnchoring<VALUE> > : public Testable<BetweenFactorWithAnchoring<VALUE> > {};

  // /**
  //  * 二元之间约束 - 强制两个值之间为给定值
  //  * 此约束要求底层类型为李类型
  //  *
  //  */
  // template<class VALUE>
  // class BetweenConstraintGiseop : public BetweenFactorWithAnchoring<VALUE> {
  // public:
  //   typedef boost::shared_ptr<BetweenConstraintGiseop<VALUE> > shared_ptr;

  //   /** 有限版本的语法糖 */
  //   BetweenConstraintGiseop(const VALUE& measured, Key key1, Key key2, double mu = 1000.0) :
  //     BetweenFactorWithAnchoring<VALUE>(key1, key2, measured,
  //                          noiseModel::Constrained::All(traits<VALUE>::GetDimension(measured), std::abs(mu)))
  //   {}

  // private:

  //   /** 序列化函数 */
  //   friend class boost::serialization::access;
  //   template<class ARCHIVE>
  //   void serialize(ARCHIVE & ar, const unsigned int /*version*/) {
  //     ar & boost::serialization::make_nvp("BetweenFactorWithAnchoring",
  //         boost::serialization::base_object<BetweenFactorWithAnchoring<VALUE> >(*this));
  //   }
  // }; // \class BetweenConstraintGiseop

  // /// traits
  // template<class VALUE>
  // struct traits<BetweenConstraintGiseop<VALUE> > : public Testable<BetweenConstraintGiseop<VALUE> > {};

} /// namespace gtsam
