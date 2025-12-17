#pragma once

// all non-specific declarations for NCIs
namespace specfem::test::fixture {

/**
 * @brief Manages views of field values along an edge.
 *
 * @tparam Initializer initializer type: must inherit
 * EdgeFieldInitializer2D::Base
 */
template <typename Initializer> struct EdgeField2D;
/**
 * @brief Initializes views of field values along an edge.
 *
 */
namespace EdgeFieldInitializer2D {
struct Base {};
template <typename AnalyticalFieldInitializer, typename EdgePointsInitializer>
struct FromAnalyticalField;
} // namespace EdgeFieldInitializer2D

/**
 * @brief Manages views of the transfer function.
 *
 * @tparam Initializer initializer type: must inherit
 * TransferFunctionInitializer2D::Base
 */
template <typename Initializer> struct TransferFunction2D;
namespace TransferFunctionInitializer2D {
struct Base {};
template <typename EdgeQuadratureInitializer,
          typename IntersectionQuadratureInitializer>
struct FromQuadratureRules;
} // namespace TransferFunctionInitializer2D

// =================================================================================================
// We may wish to move these to somewhere else in the future: these are not
// exclusive to NCIs
/**
 * @brief Manages the quadrature points and weights of a particular rule.
 *
 * @tparam Initializer initializer type: must inherit
 * QuadratureRuleInitializer::Base
 */
template <typename Initializer> struct QuadratureRule;
namespace QuadratureRuleInitializer {
struct Base {};
} // namespace QuadratureRuleInitializer

/**
 * @brief Manages 1-parameter functions. These can be used, say for
 * edge-coordinate analytical fields
 *
 * @tparam Initializer initializer type: must inherit
 * AnalyticalFieldInitializer1D::Base
 */
template <typename Initializer> struct AnalyticalField1D;
/**
 * @brief Initializes these functions.
 *
 */
namespace AnalyticalFieldInitializer1D {
struct Base {};
} // namespace AnalyticalFieldInitializer1D

/**
 * @brief Defines both an EdgeFieldInitializer2D and
 * TransferFunctionInitializer2D for an analytically known field.
 *
 * @tparam AnalyticalField Initializer for the AnalyticalField1D
 * @tparam EdgeQuadrature Initializer for the edge quadrature rule
 * @tparam IntersectionQuadrature Initializer for the intersection quadrature
 * rule
 */
template <typename AnalyticalField, typename EdgeQuadrature,
          typename IntersectionQuadrature>
struct AnalyticalFieldTransfer2D;

} // namespace specfem::test::fixture
