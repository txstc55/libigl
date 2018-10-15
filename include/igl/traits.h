#include <type_traits>

#ifndef TRAITS_H
#define TRAITS_H

namespace igl {

template <typename T>
struct is_index_scalar {
  static const bool value = std::is_same<T, std::int32_t>::value || std::is_same<T, std::int64_t>::value;
};

template <typename T>
struct is_index_matrix {
  static const bool value = is_index_scalar<typename T::Scalar>::value;
};

template <typename T>
struct is_column_vector {
  static const bool value = T::ColsAtCompileTime == 1;
};

} // namespace igl

#endif // TRAITS_H
