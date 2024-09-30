/*
SPDX-FileCopyrightText: (c) 2024, sabonerune
SPDX-License-Identifier: BSD-2-Clause
*/

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>  // NOLINT(misc-include-cleaner)
#include <world/cheaptrick.h>
#include <world/d4c.h>
#include <world/dio.h>
#include <world/harvest.h>
#include <world/stonemask.h>
#include <world/synthesis.h>

#include <climits>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

template <size_t N>
using inputNDarray = nb::ndarray<const double, nb::ndim<N>>;
template <size_t N>
using outputNDarray = nb::ndarray<nb::numpy, double, nb::ndim<N>>;

namespace {
std::mutex the_world;
template <typename T, typename U>
auto make_ndarray(std::unique_ptr<U[]>&& ptr,
                  std::initializer_list<size_t> shape) -> T {
  T out = T(ptr.get(), shape,
            nb::capsule(ptr.get(), [](void* p) noexcept { delete[] (U*)p; }));
  ptr.release();
  return out;
}

auto make_empty_ndarray() {
  return nb::ndarray<nb::numpy, double, nb::ndim<1>>(nullptr, {0},
                                                     nb::handle());
}

void validate_x_lenth(size_t x_lenth) {
  if (x_lenth > INT_MAX) {
    auto msg = "length of x must be less than or equal to " +
               std::to_string(INT_MAX) + ".";
    throw std::range_error(msg);
  }
}

void validate_samplerate(int samplerate) {
  if (samplerate <= 0) {
    throw std::invalid_argument("samplerate must be non-negative.");
  }
}

auto cheaptrick(const inputNDarray<1>& x,
                const int fs,
                const inputNDarray<1>& temporal_positions,
                const inputNDarray<1>& f0,
                const std::optional<double> q1,
                const std::optional<double> f0_floor,
                const std::optional<int> fft_size) {
  validate_x_lenth(x.size());
  validate_samplerate(fs);
  if (temporal_positions.size() != f0.size()) {
    throw std::invalid_argument(
        "The lengths of temporal_positions and f0 do not match.");
  }
  CheapTrickOption option = {};
  InitializeCheapTrickOption(fs, &option);
  if (q1) {
    option.q1 = *q1;
  }
  if (fft_size) {
    if (f0_floor) {
      const auto gil = nb::gil_scoped_acquire();
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
  } else if (f0_floor) {
    if (*f0_floor <= 0.0) {
      throw std::invalid_argument("f0_floor must be non-negative.");
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
    const auto gil = nb::gil_scoped_acquire();
    return nb::make_tuple(
        outputNDarray<2>(nullptr, {0, spectrogram_length}, nb::handle()),
        option.fft_size);
  }
  auto spectrogram = std::make_unique<double*[]>(f0_length);
  auto output_array =
      std::make_unique<double[]>(f0_length * spectrogram_length);
  for (size_t i = 0; i < f0_length; i++) {
    spectrogram[i] = &output_array[i * spectrogram_length];
  }
  {
    const std::lock_guard<std::mutex> lock(the_world);
    CheapTrick(x.data(), static_cast<int>(x.size()), fs,
               temporal_positions.data(), f0.data(),
               static_cast<int>(f0_length), &option, spectrogram.get());
  }
  {
    const auto gil = nb::gil_scoped_acquire();
    const auto result = make_ndarray<outputNDarray<2>>(
        std::move(output_array), {f0_length, spectrogram_length});
    return nb::make_tuple(result, option.fft_size);
  }
}

auto d4c(const inputNDarray<1>& x,
         const int fs,
         const inputNDarray<1>& temporal_positions,
         const inputNDarray<1>& f0,
         const int fft_size,
         const std::optional<double> threshold) {
  validate_x_lenth(x.size());
  validate_samplerate(fs);
  if (temporal_positions.size() != f0.size()) {
    throw std::invalid_argument(
        "The lengths of temporal_positions and f0 do not match.");
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
    const auto gil = nb::gil_scoped_acquire();
    return outputNDarray<2>(nullptr, {0, aperiodicity_length}, nb::handle());
  }
  const auto aperiodicity = std::make_unique<double*[]>(f0_length);
  auto output_array =
      std::make_unique<double[]>(f0_length * aperiodicity_length);
  for (size_t i = 0; i < f0_length; i++) {
    aperiodicity[i] = &output_array[i * aperiodicity_length];
  }
  {
    const std::lock_guard<std::mutex> lock(the_world);
    D4C(x.data(), static_cast<int>(x.size()), fs, temporal_positions.data(),
        f0.data(), static_cast<int>(f0_length), fft_size, &option,
        aperiodicity.get());
  }
  {
    const auto gil = nb::gil_scoped_acquire();
    return make_ndarray<outputNDarray<2>>(std::move(output_array),
                                          {f0_length, aperiodicity_length});
  }
}

auto dio(const inputNDarray<1>& x,
         const int fs,
         const std::optional<double> f0_floor,
         const std::optional<double> f0_ceil,
         const std::optional<double> channels_in_octave,
         const std::optional<double> frame_period,
         const std::optional<int> speed,
         const std::optional<double> allowed_range) {
  const size_t x_length = x.size();
  validate_x_lenth(x_length);
  validate_samplerate(fs);
  DioOption option = {};
  InitializeDioOption(&option);
  if (f0_floor) {
    option.f0_floor = *f0_floor;
  }
  if (f0_ceil) {
    option.f0_ceil = *f0_ceil;
  }
  if (channels_in_octave) {
    option.channels_in_octave = *channels_in_octave;
  }
  if (frame_period) {
    if (*frame_period <= 0) {
      throw std::invalid_argument("frame_period must be non-negative.");
    }
    option.frame_period = *frame_period;
  }
  if (speed) {
    const auto speed_max = 12;
    if (*speed <= 0 || *speed > speed_max) {
      throw std::invalid_argument("speed must be in the range 1 to 12.");
    }
    option.speed = *speed;
  }
  if (allowed_range) {
    if (*allowed_range < 0) {
      throw std::invalid_argument("allowed_range must be non-negative.");
    }
    option.allowed_range = *allowed_range;
  }
  if (x_length == 0) {
    const auto gil = nb::gil_scoped_acquire();
    return nb::make_tuple(make_empty_ndarray(), make_empty_ndarray(),
                          option.frame_period);
  }
  const size_t f0_length =
      GetSamplesForDIO(fs, static_cast<int>(x_length), option.frame_period);
  auto temporal_positions = std::make_unique<double[]>(f0_length);
  auto f0 = std::make_unique<double[]>(f0_length);
  Dio(x.data(), static_cast<int>(x_length), fs, &option,
      temporal_positions.get(), f0.get());
  {
    const auto gil = nb::gil_scoped_acquire();
    return nb::make_tuple(
        make_ndarray<outputNDarray<1>>(std::move(temporal_positions),
                                       {f0_length}),
        make_ndarray<outputNDarray<1>>(std::move(f0), {f0_length}),
        option.frame_period);
  }
}

auto harvest(const inputNDarray<1>& x,
             const int fs,
             const std::optional<double> f0_floor,
             const std::optional<double> f0_ceil,
             const std::optional<double> frame_period) {
  const size_t x_length = x.size();
  validate_x_lenth(x_length);
  validate_samplerate(fs);
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
    const auto gil = nb::gil_scoped_acquire();
    return nb::make_tuple(make_empty_ndarray(), make_empty_ndarray(),
                          option.frame_period);
  }
  const size_t f0_length =
      GetSamplesForHarvest(fs, static_cast<int>(x_length), option.frame_period);
  auto temporal_positions = std::make_unique<double[]>(f0_length);
  auto f0 = std::make_unique<double[]>(f0_length);
  Harvest(x.data(), static_cast<int>(x_length), fs, &option,
          temporal_positions.get(), f0.get());
  {
    const auto gil = nb::gil_scoped_acquire();
    return nb::make_tuple(
        make_ndarray<outputNDarray<1>>(std::move(temporal_positions),
                                       {f0_length}),
        make_ndarray<outputNDarray<1>>(std::move(f0), {f0_length}),
        option.frame_period);
  }
}

auto stonemask(const inputNDarray<1>& x,
               const int fs,
               const inputNDarray<1>& temporal_positions,
               const inputNDarray<1>& f0) {
  validate_x_lenth(x.size());
  validate_samplerate(fs);
  if (temporal_positions.size() != f0.size()) {
    throw std::invalid_argument(
        "The lengths of temporal_positions and f0 do not match.");
  }
  const size_t f0_length = f0.size();
  if (f0_length == 0) {
    const auto gil = nb::gil_scoped_acquire();
    return make_empty_ndarray();
  }
  auto refined_f0 = std::make_unique<double[]>(f0_length);
  StoneMask(x.data(), static_cast<int>(x.size()), fs, temporal_positions.data(),
            f0.data(), static_cast<int>(f0_length), refined_f0.get());
  {
    const auto gil = nb::gil_scoped_acquire();
    return make_ndarray<outputNDarray<1>>(std::move(refined_f0), {f0_length});
  }
}

auto synthesis(const inputNDarray<1>& f0,
               const inputNDarray<2>& spectrogram,
               const inputNDarray<2>& aperiodicity,
               const int fft_size,
               const double frame_period,
               const int fs) {
  validate_samplerate(fs);
  if (spectrogram.size() != aperiodicity.size()) {
    throw std::invalid_argument(
        "The lengths of spectrogram and aperiodicity do not match.");
  }
  const size_t f0_length = f0.size();
  if (spectrogram.shape(0) != f0_length) {
    throw std::invalid_argument(
        "The lengths of spectrogram and f0 do not match.");
  }
  if (aperiodicity.shape(0) != f0_length) {
    throw std::invalid_argument(
        "The lengths of aperiodicity and f0 do not match.");
  }
  const auto y_length = static_cast<size_t>(
      ((static_cast<double>(f0_length) - 1) * frame_period / 1000.0 * fs) + 1);
  if (f0_length == 0 || y_length == 0) {
    const auto gil = nb::gil_scoped_acquire();
    return make_empty_ndarray();
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
  {
    std::lock_guard<std::mutex> const lock(the_world);
    Synthesis(f0.data(), static_cast<int>(f0_length), tmp_spectram.get(),
              tmp_aperiodicity.get(), fft_size, frame_period, fs,
              static_cast<int>(y_length), y.get());
  }
  {
    const auto gil = nb::gil_scoped_acquire();
    return make_ndarray<outputNDarray<1>>(std::move(y), {y_length});
  }
}
}  // namespace

// NOLINTNEXTLINE
NB_MODULE(wwopy_ext, m) {
  m.def("cheaptrick", &cheaptrick, "x"_a, "fs"_a, "temporal_positions"_a,
        "f0"_a, "q1"_a = nb::none(), "f0_floor"_a = nb::none(),
        "fft_size"_a = nb::none(), nb::call_guard<nb::gil_scoped_release>(),
        R"(
        Calculates the spectrogram that consists of spectral envelopes.
        
        Parameters
        ----------
        x : np.NDArray[np.double]
            Input signal
        fs : int
            Sampling frequency
        temporal_positions : np.NDArray[np.double]
            Time axis
        f0 : np.NDArray[np.double]
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
        spectrogram : np.NDArray[np.double]
            Spectrogram estimated by CheapTrick.
        fft_size: int
            Automatically determined fft_size.
        
        Examples
        --------
        >>> temporal_positions, f0, frame_period = wwopy.harvest(x, fs)
        >>> spectrogram, fft_size = wwopy.cheaptrick(x, fs, temporal_positions, f0))");

  m.def("d4c", &d4c, "x"_a, "fs"_a, "temporal_positions"_a, "f0"_a,
        "fft_size"_a, "threshold"_a = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"(
        Calculates the aperiodicity.
        
        Parameters
        ----------
        x : np.NDArray[np.double]
            Input signal
        fs : int
            Sampling frequency
        temporal_positions : np.NDArray[np.double]
            Time axis
        f0 : np.NDArray[np.double]
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
        aperiodicity : np.NDArray[np.double]
            Aperiodicity estimated by D4C.
        
        Examples
        --------
        >>> temporal_positions, f0, frame_period = wwopy.harvest(x, fs)
        >>> spectrogram, fft_size = wwopy.cheaptrick(x, fs, temporal_positions, f0)
        >>> aperiodicity = wwopy.d4c(x, fs, temporal_positions, f0, fft_size))");

  m.def("dio", &dio, "x"_a, "fs"_a, "f0_floor"_a = nb::none(),
        "f0_ceil"_a = nb::none(), "channels_in_octave"_a = nb::none(),
        "frame_period"_a = nb::none(), "speed"_a = nb::none(),
        "allowed_range"_a = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"(
        Calculates the F0 contour.
        
        Parameters
        ----------
        x : np.NDArray[np.double]
            Input signal
        fs : int
            Sampling frequency
        f0_floor : float, optional
        f0_ceil : float, optional
        channels_in_octave : float, optional
        frame_period : float, optional
            Frame shift
        speed : int, optional
            Valuable speed represents the ratio for downsampling.
            The signal is downsampled to fs / speed Hz.
        allowed_range : float, optional
            Threshold used for fixing the F0 contour.
        
        Returns
        -------
        temporal_positions : np.NDArray[np.double]
            Time axis estimated by DIO.
        f0 : np.NDArray[np.double]
            F0 contour estimated by DIO.
        frame_period : float
            Automatically determined frame_period.

        Examples
        --------
        >>> temporal_positions, f0, frame_period = wwopy.dio(x, fs))");

  m.def("harvest", &harvest, "x"_a, "fs"_a, "f0_floor"_a = nb::none(),
        "f0_ceil"_a = nb::none(), "frame_period"_a = nb::none(),
        nb::call_guard<nb::gil_scoped_release>(),
        R"(
        Calculates the F0 contour.
        
        Parameters
        ----------
        x : np.NDArray[np.double]
            Input signal
        fs : int
            Sampling frequency
        f0_floor : float, optional
        f0_ceil : float, optional
        frame_period : float, optional
            Frame shift
        
        Returns
        -------
        temporal_positions : np.NDArray[np.double]
            Time axis estimated by Harvest.
        f0 : np.NDArray[np.double]
            F0 contour estimated by Harvest.
        frame_period : float
            Automatically determined frame_period.

        Examples
        --------
        >>> temporal_positions, f0, frame_period = wwopy.harvest(x, fs))");

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

  m.def("synthesis", &synthesis, "f0"_a, "spectrogram"_a, "aperiodicity"_a,
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
        >>> y = wwopy.synthesis(refined_f0, spectrogram, aperiodicity, fft_size, frame_period, fs))");
}
