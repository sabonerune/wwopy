/*
SPDX-FileCopyrightText: (c) 2024, sabonerune
SPDX-License-Identifier: BSD-2-Clause
*/

#include "wwopy_init.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>  // NOLINT(misc-include-cleaner)
#include <world/harvest.h>

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "util.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

auto harvest(
    const util::inputNDarray<1>& x,
    const int fs,
    const std::optional<double> f0_floor,
    const std::optional<double> f0_ceil,
    const std::optional<double> frame_period
) {
  const size_t x_length = x.size();
  util::validate_x_lenth(x_length);
  util::validate_fs(fs);
  HarvestOption option = {};
  InitializeHarvestOption(&option);
  if (f0_floor) {
    option.f0_floor = *f0_floor;
  }
  if (f0_ceil) {
    option.f0_ceil = *f0_ceil;
  }
  if (frame_period) {
    if (*frame_period <= 0) {
      throw std::invalid_argument("frame_period must be non-negative.");
    }
    option.frame_period = *frame_period;
  }
  if (x_length == 0) {
    const nb::gil_scoped_acquire gil;
    return nb::make_tuple(
        util::make_empty_ndarray(), util::make_empty_ndarray(),
        option.frame_period
    );
  }
  const size_t f0_length =
      GetSamplesForHarvest(fs, static_cast<int>(x_length), option.frame_period);
  auto temporal_positions = std::make_unique<double[]>(f0_length);
  auto f0 = std::make_unique<double[]>(f0_length);
  Harvest(
      x.data(), static_cast<int>(x_length), fs, &option,
      temporal_positions.get(), f0.get()
  );
  {
    const nb::gil_scoped_acquire gil;
    return nb::make_tuple(
        util::make_ndarray<util::outputNDarray<1>>(
            std::move(temporal_positions), {f0_length}
        ),
        util::make_ndarray<util::outputNDarray<1>>(std::move(f0), {f0_length}),
        option.frame_period
    );
  }
}

}  // namespace

void harvest_init(nb::module_& m) {
  m.def(
      "harvest", &harvest, "x"_a, "fs"_a, "f0_floor"_a = nb::none(),
      "f0_ceil"_a = nb::none(), "frame_period"_a = nb::none(),
      nb::call_guard<nb::gil_scoped_release>(), R"(
      Calculates the F0 contour.

      Parameters
      ----------
      x : np.ndarray[tuple[int], np.dtype[np.double]]
          Input signal
      fs : int
          Sampling frequency
      f0_floor : float, optional
      f0_ceil : float, optional
      frame_period : float, optional
          Frame shift

      Returns
      -------
      temporal_positions : np.ndarray[tuple[int], np.dtype[np.double]]
          Time axis estimated by Harvest.
      f0 : np.ndarray[tuple[int], np.dtype[np.double]]
          F0 contour estimated by Harvest.
      frame_period : float
          Automatically determined frame_period.

      Examples
      --------
      >>> temporal_positions, f0, frame_period = wwopy.harvest(x, fs))"
  );
}
