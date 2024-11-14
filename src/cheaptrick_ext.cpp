/*
SPDX-FileCopyrightText: (c) 2024, sabonerune
SPDX-License-Identifier: BSD-2-Clause
*/

#include "wwopy_init.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>  // NOLINT(misc-include-cleaner)
#include <world/cheaptrick.h>

#include <climits>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "util.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

auto cheaptrick(
    const util::inputNDarray<1>& x,
    const int fs,
    const util::inputNDarray<1>& temporal_positions,
    const util::inputNDarray<1>& f0,
    const std::optional<double> q1,
    const std::optional<double> f0_floor,
    const std::optional<int> fft_size
) {
  util::validate_x_lenth(x.size());
  util::validate_fs(fs);
  if (temporal_positions.size() != f0.size()) {
    throw std::invalid_argument(
        "The lengths of temporal_positions and f0 do not match."
    );
  }
  CheapTrickOption option{};
  InitializeCheapTrickOption(fs, &option);
  if (q1) {
    option.q1 = *q1;
  }
  if (fft_size) {
    if (f0_floor) {
      const nb::gil_scoped_acquire gil;
      const nb::object warn = nb::module_::import_("warnings").attr("warn");
      const nb::object runtimeWarning =
          nb::module_::import_("builtins").attr("RuntimeWarning");
      const auto* const msg =
          "The value of f0_floor is ignored "
          "because the value of fft_size is set.";
      warn(msg, runtimeWarning);
    }
    option.fft_size = *fft_size;
    option.f0_floor = GetF0FloorForCheapTrick(fs, *fft_size);
    if (option.f0_floor <= 0) {
      throw std::invalid_argument("fft_size is invalid.");
    }
  } else if (f0_floor) {
    if (*f0_floor <= 0.0) {
      throw std::invalid_argument("f0_floor must be non-negative.");
    }
    if (*f0_floor < GetF0FloorForCheapTrick(fs, INT_MAX)) {
      throw std::invalid_argument("Determine fft_size is invalid.");
    }
    option.f0_floor = *f0_floor;
    option.fft_size = GetFFTSizeForCheapTrick(fs, &option);
  }
  if (option.fft_size <= 0) {
    throw std::invalid_argument("fft_size must be non-negative.");
  }
  const size_t f0_length = f0.size();
  const size_t spectrogram_length = (option.fft_size / 2) + 1;
  if (f0_length == 0) {
    const nb::gil_scoped_acquire gil;
    return nb::make_tuple(
        util::outputNDarray<2>(nullptr, {0, spectrogram_length}, nb::handle()),
        option.fft_size
    );
  }
  auto spectrogram = std::make_unique<double*[]>(f0_length);
  auto output_array =
      std::make_unique<double[]>(f0_length * spectrogram_length);
  for (size_t i = 0; i < f0_length; i++) {
    spectrogram[i] = &output_array[i * spectrogram_length];
  }
  CheapTrick(
      x.data(), static_cast<int>(x.size()), fs, temporal_positions.data(),
      f0.data(), static_cast<int>(f0_length), &option, spectrogram.get()
  );
  {
    const nb::gil_scoped_acquire gil;
    const auto result = util::make_ndarray<util::outputNDarray<2>>(
        std::move(output_array), {f0_length, spectrogram_length}
    );
    return nb::make_tuple(result, option.fft_size);
  }
}

auto get_fft_size_from_f0_floor(
    const int fs,
    const std::optional<double> f0_floor
) {
  util::validate_fs(fs);
  CheapTrickOption option{};
  InitializeCheapTrickOption(fs, &option);
  if (f0_floor) {
    if (*f0_floor <= 0.0) {
      throw std::invalid_argument("f0_floor must be non-negative.");
    }
    if (*f0_floor < GetF0FloorForCheapTrick(fs, INT_MAX)) {
      throw std::invalid_argument("Determine fft_size is invalid.");
    }
    option.f0_floor = *f0_floor;
  }
  return GetFFTSizeForCheapTrick(fs, &option);
}

}  // namespace

void cheeptrick_init(nb::module_& m) {
  m.def(
      "cheaptrick", &cheaptrick, "x"_a, "fs"_a, "temporal_positions"_a, "f0"_a,
      "q1"_a = nb::none(), "f0_floor"_a = nb::none(), "fft_size"_a = nb::none(),
      nb::call_guard<nb::gil_scoped_release>(), R"(
      Calculates the spectrogram that consists of spectral envelopes.

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
      q1 : float, optional
          Used for the spectral recovery
          Since The parameter is optimized, you don't need to change the parameter.
      f0_floor : float, optional
          Whenever f0 is below this threshold the spectrum will be analyzed as if the frame is unvoiced
          We strongly recommend not to change this value unless you have enough
          knowledge of the signal processing in CheapTrick.
          Used to determine fft_size.
      fft_size : int, optional
          FFT size
          This variable has precedence over f0_floor.

      Returns
      -------
      spectrogram : np.ndarray[tuple[int, int], np.dtype[np.double]]
          Spectrogram estimated by CheapTrick.
      fft_size: int
          Automatically determined fft_size.

      Examples
      --------
      >>> temporal_positions, f0, frame_period = wwopy.harvest(x, fs)
      >>> spectrogram, fft_size = wwopy.cheaptrick(x, fs, temporal_positions, f0))"
  );
  m.def(
      "get_fft_size_from_f0_floor", &get_fft_size_from_f0_floor, "fs"_a,
      "f0_floor"_a = nb::none(), nb::call_guard<nb::gil_scoped_release>(),
      R"(
        Determine fft_size from f0_floor.

        Parameters
        ----------
        fs : int
            Sampling frequency
        f0_floor : float, optional

        Returns
        -------
        int
            Determined fft_size.
        
        Examples
        --------
        >>> temporal_positions, f0, frame_period = wwopy.harvest(x, fs)
        >>> determine_fft_size = wwopy.get_fft_size_from_f0_floor(fs, 71.0)
        >>> spectrogram, _ = wwopy.cheaptrick(x, fs, temporal_positions, f0, fft_size=determine_fft_size))"
  );
};
