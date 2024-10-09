/*
SPDX-FileCopyrightText: (c) 2024, sabonerune
SPDX-License-Identifier: BSD-2-Clause
*/

#include "wwopy_init.hpp"

#include <nanobind/nanobind.h>

#include <world/stonemask.h>

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <utility>

#include "util.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

auto stonemask(const util::inputNDarray<1>& x,
               const int fs,
               const util::inputNDarray<1>& temporal_positions,
               const util::inputNDarray<1>& f0) {
  util::validate_x_lenth(x.size());
  util::validate_fs(fs);
  if (temporal_positions.size() != f0.size()) {
    throw std::invalid_argument(
        "The lengths of temporal_positions and f0 do not match.");
  }
  const size_t f0_length = f0.size();
  if (f0_length == 0) {
    const nb::gil_scoped_acquire gil;
    return util::make_empty_ndarray();
  }
  auto refined_f0 = std::make_unique<double[]>(f0_length);
  StoneMask(x.data(), static_cast<int>(x.size()), fs, temporal_positions.data(),
            f0.data(), static_cast<int>(f0_length), refined_f0.get());
  {
    const nb::gil_scoped_acquire gil;
    return util::make_ndarray<util::outputNDarray<1>>(std::move(refined_f0), {f0_length});
  }
}

}  // namespace

void stonemask_init(nb::module_& m) {
  m.def("stonemask", &stonemask, "x"_a, "fs"_a, "temporal_positions"_a, "f0"_a,
        nb::call_guard<nb::gil_scoped_release>(),
        R"(
        Refines the estimated F0 by Dio()
        
        Parameters
        ----------
        x : np.NDArray[np.double]
            Input signal
        fs : int
            Sampling frequency
        temporal_positions : np.NDArray[np.double]
            Time axis by dio()
        f0 : np.NDArray[np.double]
            F0 contour by dio()
        
        Returns
        -------
        refined_f0 : np.NDArray[np.double]
            Refined F0.
        
        Examples
        --------
        >>> temporal_positions, f0, frame_period = wwopy.dio(x, fs)
        >>> refined_f0 = wwopy.stonemask(x, fs, temporal_positions, f0))");
}
