/*
SPDX-FileCopyrightText: (c) 2024, sabonerune
SPDX-License-Identifier: BSD-2-Clause
*/

#ifndef WWOPY_SRC_UTIL_HPP_
#define WWOPY_SRC_UTIL_HPP_

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <initializer_list>
#include <memory>

namespace util {

template <size_t N>
using inputNDarray = nanobind::ndarray<const double, nanobind::ndim<N>>;

template <size_t N>
using outputNDarray =
    nanobind::ndarray<nanobind::numpy, double, nanobind::ndim<N>>;

template <typename T, typename U>
auto make_ndarray(
    std::unique_ptr<U[]>&& ptr,
    std::initializer_list<size_t> shape
) -> T {
  T out =
      T(ptr.get(), shape,
        nanobind::capsule(ptr.get(), [](void* p) noexcept { delete[] (U*)p; }));
  ptr.release();
  return out;
}

auto make_empty_ndarray()
    -> nanobind::ndarray<nanobind::numpy, double, nanobind::ndim<1>>;

void validate_x_lenth(size_t x_lenth);
void validate_fs(int fs);
auto restore_fft_size(size_t lenth) -> int;

}  // namespace util

#endif
