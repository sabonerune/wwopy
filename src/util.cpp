/*
SPDX-FileCopyrightText: (c) 2024, sabonerune
SPDX-License-Identifier: BSD-2-Clause
*/

#include "util.hpp"

#include <nanobind/ndarray.h>

#include <climits>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

auto util::make_empty_ndarray()
    -> nb::ndarray<nanobind::numpy, double, nanobind::ndim<1>> {
  return nb::ndarray<nb::numpy, double, nb::ndim<1>>(nullptr, {0},
                                                     nb::handle());
}

void util::validate_x_lenth(size_t x_lenth) {
  if (x_lenth > INT_MAX) {
    auto msg = "length of x must be less than or equal to " +
               std::to_string(INT_MAX) + ".";
    throw std::range_error(msg);
  }
}

void util::validate_fs(int fs) {
  if (fs <= 0) {
    throw std::invalid_argument("samplerate must be non-negative.");
  }
}
