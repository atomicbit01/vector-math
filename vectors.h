

/**
 * Author: Emmanuel J
 * Date : 30-10-2026
 * Description: Math vector API
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <type_traits>
#include <utility>

namespace Math {

/**
 * @brief A fixed-size mathematical vector.
 *
 * Stores N components of type T and provides arithmetic,
 * geometric operations, comparison, and component access.
 *
 * @tparam T Component type, such as int, float, or double.
 * @tparam N Number of dimensions. Must be greater than zero.
 *
 * @note Signed integer overflow causes undefined behavior.
 * @note Division by zero follows native C++ arithmetic rules.
 * @note Floating-point calculations may introduce rounding errors.
 *
 * @example
 * Math::Vector<float, 3> position{1.0f, 2.0f, 3.0f};
 * Math::Vector<float, 3> velocity{0.0f, 1.0f, 0.0f};
 * position += velocity;
 */
template <typename T, std::size_t N> struct Vector {
  static_assert(N > 0, "Vector dimension size must be greater than 0");

private:
  T dim[N];

public:
  /**
   * @brief Adds another vector component by component.
   *
   * @tparam X Component type of the other vector.
   * @param vector Vector to add.
   * @return A new vector whose component type is deduced from
   *         the addition of T and X.
   *
   * @note The original vector is not modified.
   */
  template <typename X>
  constexpr auto operator+(const Vector<X, N> &vector) const {
    using Result = decltype(std::declval<T>() + std::declval<X>());

    Vector<Result, N> result{};

    for (std::size_t i = 0; i < N; ++i) {
      result.dim[i] =
          static_cast<Result>(dim[i]) + static_cast<Result>(vector.dim[i]);
    }

    return result;
  }

  /**
   * @brief Subtracts another vector component by component.
   *
   * @tparam X Component type of the other vector.
   * @param vector Vector to subtract.
   * @return A new vector whose component type is deduced from
   *         the subtraction of T and X.
   *
   * @note The original vector is not modified.
   */
  template <typename X>
  constexpr auto operator-(const Vector<X, N> &vector) const {
    using Result = decltype(std::declval<T>() - std::declval<X>());

    Vector<Result, N> result{};

    for (std::size_t i = 0; i < N; ++i) {
      result.dim[i] =
          static_cast<Result>(dim[i]) - static_cast<Result>(vector.dim[i]);
    }

    return result;
  }

  /**
   * @brief Computes the dot product of two vectors.
   *
   * Multiplies corresponding components and sums their products.
   *
   * @tparam X Component type of the other vector.
   * @param vector Vector to compute the dot product with.
   * @return The scalar dot product.
   *
   * @note The result type is deduced from component multiplication
   *       and addition.
   * @note Integer overflow may cause undefined behavior.
   */
  template <typename X>
  constexpr auto operator*(const Vector<X, N> &vector) const {
    using Product = decltype(std::declval<T>() * std::declval<X>());
    using Result = decltype(std::declval<Product>() + std::declval<Product>());

    Result result{};

    for (std::size_t i = 0; i < N; ++i) {
      result += dim[i] * vector.dim[i];
    }

    return result;
  }

  /**
   * @brief Multiplies every component by a scalar.
   *
   * @tparam X Scalar type.
   * @param scalar Value to multiply each component by.
   * @return A new vector with the deduced multiplication result type.
   *
   * @note The original vector is not modified.
   */
  template <typename X> constexpr auto operator*(const X &scalar) const {
    using Result = decltype(std::declval<T>() * std::declval<X>());

    Vector<Result, N> result{};

    for (std::size_t i = 0; i < N; ++i) {
      result.dim[i] = dim[i] * scalar;
    }

    return result;
  }

  /**
   * @brief Divides every component by a scalar.
   *
   * @tparam X Scalar type.
   * @param scalar Divisor.
   * @return A new vector with the deduced division result type.
   *
   * @note Integer division truncates fractional results.
   * @note Integer division by zero causes undefined behavior.
   * @note Floating-point division by zero may produce infinity or NaN.
   */
  template <typename X> constexpr auto operator/(const X &scalar) const {
    using Result = decltype(std::declval<T>() / std::declval<X>());

    Vector<Result, N> result{};

    for (std::size_t i = 0; i < N; ++i) {
      result.dim[i] = dim[i] / scalar;
    }

    return result;
  }

  /**
   * @brief Calculates the Euclidean length (magnitude) of the vector.
   *
   * @return The length as a double.
   *
   * @note Uses std::hypot to improve numerical stability for very
   *       large or very small floating-point components.
   * @note Converting components to double may lose precision for
   *       some large integer values.
   */
  constexpr auto length() const {
    double magnitude = 0.0;

    for (std::size_t i = 0; i < N; ++i) {
      magnitude = std::hypot(magnitude, static_cast<double>(dim[i]));
    }

    return magnitude;
  }

  /**
   * @brief Returns a unit vector in the same direction.
   *
   * @return A normalized vector whose component type is deduced
   *         from dividing T by double.
   *
   * @note A zero vector returns a zero vector.
   * @note The resulting length is approximately 1 for nonzero vectors.
   * @note Floating-point rounding may affect the result.
   */
  constexpr auto normalize() const {
    auto magnitude = this->length();

    using Result =
        decltype(std::declval<T>() / std::declval<decltype(magnitude)>());

    Vector<Result, N> result{};

    if (magnitude == 0) {
      return result;
    }

    for (std::size_t i = 0; i < N; ++i) {
      result.dim[i] = dim[i] / magnitude;
    }

    return result;
  }

  /**
   * @brief Computes the cross product of two 3D vectors.
   *
   * @tparam X Component type of the other vector.
   * @tparam M Internal dimension constraint used for SFINAE.
   * @param vector The other 3D vector.
   * @return A new 3D vector perpendicular to both input vectors.
   *
   * @note Available only when N is 3.
   * @note The result follows the right-hand rule.
   */
  template <typename X, std::size_t M = N,
            typename std::enable_if<M == N && N == 3, int>::type = 0>
  constexpr auto cross(const Vector<X, 3> &vector) const {
    using Product = decltype(std::declval<T>() * std::declval<X>());
    using Result = decltype(std::declval<Product>() - std::declval<Product>());

    Vector<Result, 3> result{};

    const auto a0 = static_cast<Result>(dim[0]);
    const auto a1 = static_cast<Result>(dim[1]);
    const auto a2 = static_cast<Result>(dim[2]);

    const auto b0 = static_cast<Result>(vector.dim[0]);
    const auto b1 = static_cast<Result>(vector.dim[1]);
    const auto b2 = static_cast<Result>(vector.dim[2]);

    result.dim[0] = a1 * b2 - a2 * b1;
    result.dim[1] = a2 * b0 - a0 * b2;
    result.dim[2] = a0 * b1 - a1 * b0;

    return result;
  }

  /**
   * @brief Calculates the distance between this vector and another.
   *
   * @tparam X Component type of the other vector.
   * @param vector The other vector.
   * @return The Euclidean distance as a double.
   */
  template <typename X>
  constexpr auto distanceFrom(const Vector<X, N> &vector) const {
    return (*this - vector).length();
  }

  /**
   * @brief Calculates the angle between two vectors in radians.
   *
   * @tparam X Component type of the other vector.
   * @param vector The other vector.
   * @return The angle in radians, in the range [0, pi].
   *
   * @note Returns 0.0 if either vector has zero length.
   * @note The cosine is clamped to [-1, 1] to help avoid floating-point
   *       rounding errors before calling std::acos.
   */
  template <typename X>
  constexpr auto angleBetween(const Vector<X, N> &vector) const {
    auto mag1 = length();
    auto mag2 = vector.length();

    if (mag1 == 0 || mag2 == 0) {
      return 0.0;
    }

    const double cosine = static_cast<double>((*this) * vector) / (mag1 * mag2);

    return std::acos(std::clamp(cosine, -1.0, 1.0));
  }

  /**
   * @brief Checks whether the vector has approximately unit length.
   *
   * @return True if the length differs from 1.0 by no more than 1e-5.
   */
  constexpr bool isNormalized() const {
    return std::abs(this->length() - 1.0) <= 1e-5;
  }

  /**
   * @brief Checks whether every component is approximately zero.
   *
   * @return True if every component has absolute value at most 1e-5.
   *
   * @note Requires a component type supported by std::abs.
   */
  bool isZero() const {
    for (std::size_t i = 0; i < N; ++i) {
      if (std::abs(dim[i]) > 1e-5) {
        return false;
      }
    }

    return true;
  }

  /**
   * @brief Provides mutable access to a component.
   *
   * @param index Zero-based component index.
   * @return A reference to the selected component.
   *
   * @warning No bounds checking is performed. The index must be less
   *          than N; otherwise, accessing the element is undefined behavior.
   */
  constexpr T &operator[](std::size_t index) { return dim[index]; }

  /**
   * @brief Provides read-only access to a component.
   *
   * @param index Zero-based component index.
   * @return A const reference to the selected component.
   *
   * @warning No bounds checking is performed. The index must be less
   *          than N; otherwise, accessing the element is undefined behavior.
   */
  constexpr const T &operator[](std::size_t index) const { return dim[index]; }

  /**
   * @brief Calculates the squared length of the vector.
   *
   * @return The dot product of the vector with itself.
   *
   * @note Avoids the square root used by length().
   * @note The return type follows the dot-product implementation.
   */
  constexpr auto squaredLength() const { return (*this) * (*this); }

  /**
   * @brief Constructs a vector from exactly N component values.
   *
   * @tparam Args Types of the supplied component values.
   * @param args Component values, converted to T.
   *
   * @note Enabled only when exactly N arguments are provided.
   * @note Conversions to T may lose precision or change values.
   */
  template <typename... Args,
            typename std::enable_if<sizeof...(Args) == N, int>::type = 0>
  constexpr Vector(Args... args) : dim{static_cast<T>(args)...} {}

  /**
   * @brief Constructs a vector with every component value-initialized.
   *
   * For arithmetic component types, all components become zero.
   */
  constexpr Vector() : dim{} {}

  /**
   * @brief Allows different Vector specializations to access each
   *        other's private components.
   */
  template <typename, std::size_t> friend struct Vector;

  /**
   * @brief Adds another vector to this vector in place.
   *
   * @tparam X Component type of the other vector.
   * @param vector Vector to add.
   * @return Reference to this vector.
   *
   * @note The result of each component operation is converted back to T.
   */
  template <typename X>
  constexpr Vector &operator+=(const Vector<X, N> &vector) {
    for (std::size_t i = 0; i < N; ++i) {
      dim[i] = static_cast<T>(dim[i] + vector.dim[i]);
    }

    return *this;
  }

  /**
   * @brief Subtracts another vector from this vector in place.
   *
   * @tparam X Component type of the other vector.
   * @param vector Vector to subtract.
   * @return Reference to this vector.
   *
   * @note The result of each component operation is converted back to T.
   */
  template <typename X>
  constexpr Vector &operator-=(const Vector<X, N> &vector) {
    for (std::size_t i = 0; i < N; ++i) {
      dim[i] = static_cast<T>(dim[i] - vector.dim[i]);
    }

    return *this;
  }

  /**
   * @brief Divides every component by a scalar in place.
   *
   * @tparam X Scalar type.
   * @param scalar Divisor.
   * @return Reference to this vector.
   *
   * @note Results are converted back to T.
   * @note Integer division by zero causes undefined behavior.
   */
  template <typename X> constexpr Vector &operator/=(const X &scalar) {
    for (std::size_t i = 0; i < N; ++i) {
      dim[i] = static_cast<T>(dim[i] / scalar);
    }

    return *this;
  }

  /**
   * @brief Multiplies every component by a scalar in place.
   *
   * @tparam X Scalar type.
   * @param scalar Multiplier.
   * @return Reference to this vector.
   *
   * @note Results are converted back to T.
   */
  template <typename X> constexpr Vector &operator*=(const X &scalar) {
    for (std::size_t i = 0; i < N; ++i) {
      dim[i] = static_cast<T>(dim[i] * scalar);
    }

    return *this;
  }

  /**
   * @brief Returns the negated vector.
   *
   * @return A new vector whose component type is deduced from -T.
   *
   * @note The original vector is not modified.
   */
  constexpr auto operator-() const {
    Vector<decltype(-std::declval<T>()), N> result{};

    for (std::size_t i = 0; i < N; ++i) {
      result.dim[i] = -dim[i];
    }

    return result;
  }

  /**
   * @brief Compares two vectors for exact component equality.
   *
   * @tparam X Component type of the other vector.
   * @param vector Vector to compare against.
   * @return True if all corresponding components compare equal.
   *
   * @note Floating-point components are compared exactly, not with
   *       an epsilon tolerance.
   */
  template <typename X>
  constexpr bool operator==(const Vector<X, N> &vector) const {
    for (std::size_t s = 0, e = N; s < e; ++s) {
      --e;

      if (dim[s] != vector.dim[s] || dim[e] != vector.dim[e]) {
        return false;
      }
    }

    return true;
  }

  /**
   * @brief Checks whether two vectors are not exactly equal.
   *
   * @tparam X Component type of the other vector.
   * @param vector Vector to compare against.
   * @return True if at least one pair of components differs.
   */
  template <typename X>
  constexpr bool operator!=(const Vector<X, N> &vector) const {
    return !(*this == vector);
  }
};

/**
 * @brief Multiplies a vector by a scalar on the left.
 *
 * @tparam X Scalar type.
 * @tparam T Vector component type.
 * @tparam N Vector dimension.
 * @param scalar Scalar multiplier.
 * @param vector Vector to multiply.
 * @return The same result as vector * scalar.
 */
template <typename X, typename T, std::size_t N>
constexpr auto operator*(const X &scalar, const Vector<T, N> &vector) {
  return vector * scalar;
}

/**
 * @brief Provides the positive coordinate axes for a 3D coordinate system.
 *
 * @tparam T Component type, defaulting to double.
 *
 * @note X = (1, 0, 0), Y = (0, 1, 0), Z = (0, 0, 1).
 */
template <typename T = double> struct Axis {
  /// @brief Unit vector along the positive X axis.
  static constexpr auto X = Math::Vector<T, 3>{
      static_cast<T>(1), static_cast<T>(0), static_cast<T>(0)};

  /// @brief Unit vector along the positive Y axis.
  static constexpr auto Y = Math::Vector<T, 3>{
      static_cast<T>(0), static_cast<T>(1), static_cast<T>(0)};

  /// @brief Unit vector along the positive Z axis.
  static constexpr auto Z = Math::Vector<T, 3>{
      static_cast<T>(0), static_cast<T>(0), static_cast<T>(1)};
};

/**
 * @brief Writes a vector to an output stream.
 *
 * @tparam T Vector component type.
 * @tparam N Vector dimension.
 * @param os Output stream.
 * @param vector Vector to print.
 * @return Reference to the output stream.
 *
 * @example
 * Math::Vector<int, 3> v{1, 2, 3};
 * std::cout << v; // Prints: (1, 2, 3)
 */
template <typename T, std::size_t N>
std::ostream &operator<<(std::ostream &os, const Vector<T, N> &vector) {
  os << '(';

  for (std::size_t i = 0; i < N; ++i) {
    if (i > 0) {
      os << ", ";
    }

    os << vector[i];
  }

  os << ')';
  return os;
}

} // namespace Math
