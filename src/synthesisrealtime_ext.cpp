/*
SPDX-FileCopyrightText: (c) 2024, sabonerune
SPDX-License-Identifier: BSD-2-Clause
*/

#include "wwopy_init.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>  // NOLINT(misc-include-cleaner)
#include <world/synthesisrealtime.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "util.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

class RealtimeSynthesizer {
 private:
  WorldSynthesizer synthesizer;

 public:
  RealtimeSynthesizer(
      int fs,
      double frame_period,
      int fft_size,
      int buffer_size,
      int number_of_pointers
  );
  ~RealtimeSynthesizer();
  auto append(
      const util::inputNDarray<1>& f0,
      const util::inputNDarray<2>& spectrogram,
      const util::inputNDarray<2>& aperiodicity
  ) -> bool;
  auto locked() -> bool;
  auto synthesis() -> std::optional<util::outputNDarray<1>>;
  void refresh();
};

RealtimeSynthesizer::RealtimeSynthesizer(
    const int fs,
    const double frame_period,
    const int fft_size,
    const int buffer_size,
    const int number_of_pointers
) {
  util::validate_fs(fs);
  if (frame_period <= 0) {
    throw std::invalid_argument("frame_period must be greater than 0.");
  }
  if (fft_size <= 0) {
    throw std::invalid_argument("fft_size must be greater than 0.");
  }
  if (buffer_size <= 0) {
    throw std::invalid_argument("buffer_size must be greater than 0.");
  }
  if (number_of_pointers <= 0) {
    throw std::invalid_argument("number_of_pointers must be greater than 0.");
  }
  synthesizer = {};
  InitializeSynthesizer(
      fs, frame_period, fft_size, buffer_size, number_of_pointers, &synthesizer
  );
}

RealtimeSynthesizer::~RealtimeSynthesizer() {
  DestroySynthesizer(&synthesizer);
}

auto RealtimeSynthesizer::append(
    const util::inputNDarray<1>& f0,
    const util::inputNDarray<2>& spectrogram,
    const util::inputNDarray<2>& aperiodicity
) -> bool {
  const size_t f0_length = f0.shape(0);
  if (f0_length != spectrogram.shape(0) || f0_length != aperiodicity.shape(0)) {
    throw std::invalid_argument(
        "The lengths of f0 or spectrogram or aperiodicity do not match."
    );
  }
  const size_t sp_length = spectrogram.shape(1);
  if (sp_length != aperiodicity.shape(1)) {
    throw std::invalid_argument(
        "The lengths of spectrogram and aperiodicity do not match."
    );
  }
  if ((sp_length - 1) * 2 != synthesizer.fft_size) {
    throw std::invalid_argument(
        "The lengths of spectrogram and aperiodicity do not match fft_size."
    );
  }
  if (f0_length == 0) {
    return true;
  }
  auto f0_in = std::make_unique<double[]>(f0_length);
  auto spectrogram_in = std::make_unique<double*[]>(sp_length);
  auto aperiodicity_in = std::make_unique<double*[]>(sp_length);
  const auto cleanup = [&]() noexcept -> void {
    for (size_t i = 0; i < sp_length; i++) {
      delete[] spectrogram_in[i];
      delete[] aperiodicity_in[i];
    }
  };
  try {
    {
      std::copy_n(f0.data(), f0_length, f0_in.get());
      const auto* sp_tmp = spectrogram.data();
      const auto* ap_tmp = aperiodicity.data();
      for (size_t i = 0; i < f0_length; i++) {
        auto* sp = new double[sp_length];
        auto* ap = new double[sp_length];
        std::copy_n(&sp_tmp[i * sp_length], sp_length, sp);
        std::copy_n(&ap_tmp[i * sp_length], sp_length, ap);
        spectrogram_in[i] = sp;
        aperiodicity_in[i] = ap;
      }
    }
    const bool result =
        AddParameters(
            f0_in.get(), static_cast<int>(f0_length), spectrogram_in.get(),
            aperiodicity_in.get(), &synthesizer
        ) != 0;
    if (result) {
      f0_in.release();
      spectrogram_in.release();
      aperiodicity_in.release();
    } else {
      cleanup();
    }
    return result;
  } catch (const std::exception& e) {
    cleanup();
    throw e;
  }
}

auto RealtimeSynthesizer::locked() -> bool {
  return IsLocked(&synthesizer) != 0;
}

auto RealtimeSynthesizer::synthesis() -> std::optional<util::outputNDarray<1>> {
  if (Synthesis2(&synthesizer) == 0) {
    return std::nullopt;
  }
  const auto buffer_size = static_cast<size_t>(synthesizer.buffer_size);
  auto y = std::make_unique<double[]>(buffer_size);
  std::copy_n(synthesizer.buffer, buffer_size, y.get());
  {
    const nb::gil_scoped_acquire gil;
    return util::make_ndarray<util::outputNDarray<1>>(
        std::move(y), {buffer_size}
    );
  }
}

void RealtimeSynthesizer::refresh() {
  RefreshSynthesizer(&synthesizer);
}

}  // namespace

void synthesisrealtime_init(nb::module_& m) {
  nb::class_<RealtimeSynthesizer>(m, "RealtimeSynthesizer", R"(
  RealtimeSynthesizer

  Voice synthesis based on f0, spectrogram and aperiodicity.
  This is an implementation for real-time applications.
  )")
      .def(
          nb::init<const int, const double, const int, const int, const int>(),
          "fs"_a, "frame_period"_a, "fft_size"_a, "buffer_size"_a,
          "number_of_pointers"_a, nb::call_guard<nb::gil_scoped_release>(), R"(
          Initializes the synthesizer based on basic parameters.

          Parameters
          ----------
          fs : int
              Sampling frequency
          frame_period : float
              Frame period (ms)
          fft_size : int
              FFT size
          buffer_size : int
              Buffer size (sample)
          number_of_pointers : int
              The number of elements in the ring buffer
          )"
      )
      .def(
          "append", &RealtimeSynthesizer::append,
          nb::call_guard<nb::gil_scoped_release>(), R"(
          Attempts to add speech parameters.
          You can add several frames at the same time.

          Parameters
          ----------
          f0 : np.NDArray[np.double]
              F0 contour with length of f0_length
          spectrogram : np.NDArray[np.double]
              Spectrogram
          aperiodicity : np.NDArray[np.double]
              Aperiodicity

          Returns
          -------
          bool
              True if added successfully.
              Retrun True if the parameter is an empty array.
          )"
      )
      .def(
          "locked", &RealtimeSynthesizer::locked,
          R"(
          Checks whether the synthesizer is locked or not.
          "Lock" is defined as the situation that the ring buffer cannot add parameters and cannot synthesize the waveform.
          It will be caused when the duration calculated by the number of added frames is below 1 / F0 + buffer_size / fs.
          If this function returns True, please refresh the synthesizer.

          Returns
          -------
          bool
          )"
      )
      .def(
          "synthesis", &RealtimeSynthesizer::synthesis,
          nb::call_guard<nb::gil_scoped_release>(),
          R"(
          Generates speech with length of buffer_size sample.

          Returns
          -------
          np.NDArray[np.double] or None
          )"
      )
      .def(
          "refresh", &RealtimeSynthesizer::refresh,
          nb::call_guard<nb::gil_scoped_release>(),
          "Sets the parameters to default."
      );
}
