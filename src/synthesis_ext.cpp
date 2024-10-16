/*
SPDX-FileCopyrightText: (c) 2024, sabonerune
SPDX-License-Identifier: BSD-2-Clause
*/

#include "wwopy_init.hpp"

#include <nanobind/nanobind.h>
#include <world/synthesis.h>

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <utility>

#include "util.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

auto synthesis(
    const util::inputNDarray<1>& f0,
    const util::inputNDarray<2>& spectrogram,
    const util::inputNDarray<2>& aperiodicity,
    const int fft_size,
    const double frame_period,
    const int fs
) {
  util::validate_fs(fs);
  if (spectrogram.size() != aperiodicity.size()) {
    throw std::invalid_argument(
        "The lengths of spectrogram and aperiodicity do not match."
    );
  }
  const size_t f0_length = f0.size();
  if (spectrogram.shape(0) != f0_length) {
    throw std::invalid_argument(
        "The lengths of spectrogram and f0 do not match."
    );
  }
  if (aperiodicity.shape(0) != f0_length) {
    throw std::invalid_argument(
        "The lengths of aperiodicity and f0 do not match."
    );
  }
  const auto y_length = static_cast<size_t>(
      ((static_cast<double>(f0_length) - 1) * frame_period / 1000.0 * fs) + 1
  );
  if (f0_length == 0 || y_length == 0) {
    const nb::gil_scoped_acquire gil;
    return util::make_empty_ndarray();
  }
  auto tmp_spectram = std::make_unique<const double*[]>(f0_length);
  auto tmp_aperiodicity = std::make_unique<const double*[]>(f0_length);
  {
    const double* spectrogram_data = spectrogram.data();
    const double* aperiodicity_data = aperiodicity.data();
    const size_t spectrogram_length = spectrogram.shape(1);
    const size_t aperiodicity_length = aperiodicity.shape(1);
    for (size_t i = 0; i < f0_length; i++) {
      tmp_spectram[i] = &spectrogram_data[spectrogram_length * i];
      tmp_aperiodicity[i] = &aperiodicity_data[aperiodicity_length * i];
    }
  }
  auto y = std::make_unique<double[]>(y_length);
  Synthesis(
      f0.data(), static_cast<int>(f0_length), tmp_spectram.get(),
      tmp_aperiodicity.get(), fft_size, frame_period, fs,
      static_cast<int>(y_length), y.get()
  );
  {
    const nb::gil_scoped_acquire gil;
    return util::make_ndarray<util::outputNDarray<1>>(std::move(y), {y_length});
  }
}

}  // namespace

void synthesis_init(nb::module_& m) {
  m.def(
      "synthesis", &synthesis, "f0"_a, "spectrogram"_a, "aperiodicity"_a,
      "fft_size"_a, "frame_period"_a, "fs"_a,
      nb::call_guard<nb::gil_scoped_release>(),
      R"(
        Synthesize the voice based on f0, spectrogram and aperiodicity.
        
        Parameters
        ----------
        f0 : np.NDArray[np.double]
            f0 contour
        spectrogram : np.NDArray[np.double]
            Spectrogram
        aperiodicity : np.NDArray[np.double]
            Aperiodicity spectrogram
        fft_size : int
            FFT size
        frame_period : float
            Temporal period used for the analysis
        fs : int
            Sampling frequency
        
        Returns
        -------
        y : np.NDArray[np.double]
            Calculated speech.
        
        Examples
        --------
        >>> temporal_positions, f0, frame_period = wwopy.dio(x, fs)
        >>> refined_f0 = wwopy.stonemask(x, fs, temporal_positions, f0)
        >>> spectrogram, fft_size = wwopy.cheaptrick(x, fs, temporal_positions, refined_f0)
        >>> aperiodicity = wwopy.d4c(x, fs, temporal_positions, refined_f0, fft_size)
        >>> y = wwopy.synthesis(refined_f0, spectrogram, aperiodicity, fft_size, frame_period, fs))"
  );
}
