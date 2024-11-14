/*
SPDX-FileCopyrightText: (c) 2024, sabonerune
SPDX-License-Identifier: BSD-2-Clause
*/

#include "util.hpp"

#include <nanobind/ndarray.h>

#include <climits>
#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace nb = nanobind;

auto util::make_empty_ndarray()
    -> nb::ndarray<nanobind::numpy, double, nanobind::ndim<1>> {
  return nb::ndarray<nb::numpy, double, nb::ndim<1>>(
      nullptr, {0}, nb::handle()
  );
}

void util::validate_x_lenth(size_t x_lenth) {
  if (x_lenth > INT_MAX) {
    std::basic_ostringstream<char> s;
    s << "length of x must be less than or equal to " << INT_MAX << ".";
    throw std::range_error(s.str());
  }
}

void util::validate_fs(int fs) {
  if (fs <= 0) {
    throw std::invalid_argument("samplerate must be non-negative.");
  }
}

auto util::restore_fft_size(size_t lenth) -> int {
  if (lenth < 2) {
    throw std::invalid_argument("lenth is too small");
  }
  const auto result = (lenth - 1) * 2;
  return static_cast<int>(result);
}
