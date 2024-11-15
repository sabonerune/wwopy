/*
SPDX-FileCopyrightText: (c) 2024, sabonerune
SPDX-License-Identifier: BSD-2-Clause
*/

#include "wwopy_init.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>  // NOLINT(misc-include-cleaner)
#include <world/d4c.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "util.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

auto d4c(
    const util::inputNDarray<1>& x,
    const int fs,
    const util::inputNDarray<1>& temporal_positions,
    const util::inputNDarray<1>& f0,
    const int fft_size,
    const std::optional<double> threshold
) {
  util::validate_x_lenth(x.size());
  util::validate_fs(fs);
  if (temporal_positions.size() != f0.size()) {
    throw std::invalid_argument(
        "The lengths of temporal_positions and f0 do not match."
    );
  }
  if (fft_size <= 0) {
    throw std::invalid_argument("fft_size must be non-negative.");
  }
  D4COption option = {};
  InitializeD4COption(&option);
  if (threshold) {
    option.threshold = *threshold;
  }
  const size_t f0_length = f0.size();
  const size_t aperiodicity_length = (fft_size / 2) + 1;
  if (f0_length == 0) {
    const nb::gil_scoped_acquire gil;
    return util::outputNDarray<2>(
        nullptr, {0, aperiodicity_length}, nb::handle()
    );
  }
  const auto aperiodicity = std::make_unique<double*[]>(f0_length);
  auto output_array =
      std::make_unique<double[]>(f0_length * aperiodicity_length);
  for (size_t i = 0; i < f0_length; i++) {
    aperiodicity[i] = &output_array[i * aperiodicity_length];
  }
  D4C(x.data(), static_cast<int>(x.size()), fs, temporal_positions.data(),
      f0.data(), static_cast<int>(f0_length), fft_size, &option,
      aperiodicity.get());
  {
    const nb::gil_scoped_acquire gil;
    return util::make_ndarray<util::outputNDarray<2>>(
        std::move(output_array), {f0_length, aperiodicity_length}
    );
  }
}

}  // namespace

void d4c_init(nb::module_& m) {
  m.def(
      "d4c", &d4c, "x"_a, "fs"_a, "temporal_positions"_a, "f0"_a, "fft_size"_a,
      "threshold"_a = nb::none(), nb::call_guard<nb::gil_scoped_release>(), R"(
      Calculates the aperiodicity.

      Parameters
      ----------
      x : np.ndarray[tuple[int], np.dtype[np.double]]
          Input signal
      fs : int
          Sampling frequency
      temporal_positions : np.ndarray[tuple[int], np.dtype[np.double]]
          Time axis
      f0 : np.ndarray[tuple[int], np.dtype[np.double]]
          F0 contour
      fft_size : int
          FFT size
          Typically this will be the same as DIO or Harvest.
      threshold : float, optional
          It is used to determine the aperiodicity in whole frequency band.
          D4C identifies whether the frame is voiced segment even if it had an F0.
          If the estimated value falls below the threshold,
          the aperiodicity in whole frequency band will set to 1.0.

      Returns
      -------
      np.ndarray[tuple[int, int], np.dtype[np.double]]
          Aperiodicity estimated by D4C.

      Examples
      --------
      >>> temporal_positions, f0, frame_period = wwopy.harvest(x, fs)
      >>> spectrogram, fft_size = wwopy.cheaptrick(x, fs, temporal_positions, f0)
      >>> aperiodicity = wwopy.d4c(x, fs, temporal_positions, f0, fft_size))"
  );
}
