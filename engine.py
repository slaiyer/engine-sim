#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "numpy",
#   "scipy",
#   "soundfile",
# ]
# ///

import enum
import functools
import numpy
import scipy
import soundfile
from typing import Callable, List, Tuple

firing_degrees: tuple[float, ...] = (
    0.0,
    135.0,
)
base_rpm: float = 1000.0
redline_rpm: float = 8750.0
num_gears: int = 6

idle_time: float = 3.0
redline_hold_time: float = 1
accel_time_per_gear: float = 2
decel_time_per_gear: float = 1.5

sample_path: str = "./kick.wav"
output_path: str = "./engine.wav"
sr: int = 44100
sample_duration_sec: float = 0.1


class Easing(enum.Enum):
    COSINE = "cosine"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    CUBIC = "cubic"
    QUARTIC = "quartic"
    QUINTIC = "quintic"
    SEXTIC = "sextic"
    SEPTIC = "septic"


easing_type: Easing = Easing.SEPTIC

upshift_durations = (0.3, 0.25, 0.2, 0.15, 0.1)
downshift_durations = upshift_durations[::-1]

gear_start_rpms = [base_rpm + i * 1000 for i in range(num_gears)]
gear_end_rpms = [gear_start_rpms[i] + 2250 for i in range(num_gears)]
gear_end_rpms[-1] = redline_rpm

num_cylinders = len(firing_degrees)
crank_degrees_per_cycle: float = 720.0  # 4-stroke

pulse, _ = soundfile.read(sample_path)
if pulse.ndim > 1:
    pulse = pulse.mean(axis=1)
pulse = pulse[: int(sample_duration_sec * sr)]


def bass_boost(
    data: numpy.ndarray,
    gain_db: float = 6.0,
    cutoff: float = 150.0,
    fs: int = sr,
) -> numpy.ndarray:
    gain = 10 ** (gain_db / 20)
    b, a = scipy.signal.butter(N=2, Wn=cutoff / (fs / 2), btype="low")
    low = scipy.signal.lfilter(b, a, data)
    return data + (low * (gain - 1))


def compressor(
    signal: numpy.ndarray,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    knee_db: float = 6.0,  # softer knee
    makeup_gain_db: float = 5.0,  # compensate for compression
    eps: float = 1e-10,
) -> numpy.ndarray:
    amplitude = numpy.maximum(numpy.abs(signal), eps)
    signal_db = 20 * numpy.log10(amplitude)

    def compress_sample(db: float) -> float:
        delta = db - threshold_db
        if knee_db > 0 and abs(delta) <= knee_db / 2:
            # Soft knee (quadratic interpolation)
            return db + (1 / ratio - 1) * ((delta + knee_db / 2) ** 2) / (2 * knee_db)
        elif db > threshold_db:
            return threshold_db + (db - threshold_db) / ratio
        else:
            return db

    compress_fn = numpy.vectorize(compress_sample)
    compressed_db = compress_fn(signal_db)
    compressed_amp = 10 ** ((compressed_db + makeup_gain_db) / 20)
    return numpy.sign(signal) * compressed_amp


def apply_audio_effects_pipeline(
    signal: numpy.ndarray,
    effects_pipeline: List[Tuple[Callable, dict]],
) -> numpy.ndarray:
    processed_signal = signal.copy()  # avoid modifying original
    for effect_func, params in effects_pipeline:
        processed_signal = effect_func(processed_signal, **params)
    return processed_signal


@functools.cache
def get_easing_curve(easing: Easing) -> Callable[[numpy.ndarray], numpy.ndarray]:
    match easing:
        case Easing.COSINE:
            return lambda t: numpy.sin(t * (numpy.pi / 2))
        case Easing.LINEAR:
            return lambda t: t
        case Easing.QUADRATIC:
            return lambda t: t**2
        case Easing.CUBIC:
            return lambda t: t**3
        case Easing.QUARTIC:
            return lambda t: t**4
        case Easing.QUINTIC:
            return lambda t: t**5
        case Easing.SEXTIC:
            return lambda t: t**6
        case Easing.SEPTIC:
            return lambda t: t**7
        case _:
            raise ValueError(f"unknown easing type: {easing}")


@functools.cache
def get_pulse(
    rpm: float,
    enable_pitch_change: bool = False,
) -> numpy.ndarray:
    if not enable_pitch_change:
        if len(pulse) < len(pulse):
            return numpy.pad(pulse.copy(), (0, len(pulse) - len(pulse)))
        return pulse.copy()

    factor = rpm / base_rpm
    new_len = max(16, int(len(pulse) / factor))
    pitched = scipy.signal.resample(pulse, new_len)

    alpha = numpy.clip((rpm - base_rpm) / (redline_rpm - base_rpm) * 0.3, 0, 0.3)
    window = scipy.signal.windows.tukey(len(pitched), alpha=alpha)
    pitched *= window

    pitched *= numpy.sqrt(len(pulse) / len(pitched))

    if len(pitched) < len(pulse):
        pitched = numpy.pad(pitched, (0, len(pulse) - len(pitched)))
    else:
        pitched = pitched[: len(pulse)]

    return pitched


def build_rpm_profile() -> numpy.ndarray:
    easing = get_easing_curve(easing_type)
    profile = []

    # idle hold
    profile.append(numpy.full(int(idle_time * sr), base_rpm))

    # acceleration + upshifts
    for g in range(num_gears - 1):
        start, end = gear_start_rpms[g], gear_end_rpms[g]
        accel = numpy.linspace(start, end, int(accel_time_per_gear * sr))
        profile.append(accel)

        # Upshift dip
        dip_duration = upshift_durations[g]
        t = numpy.linspace(0, 1, int(dip_duration * sr))
        dip = end - (end - gear_start_rpms[g + 1]) * easing(t)
        profile.append(dip)

    # final gear acceleration
    final_accel = numpy.linspace(gear_start_rpms[-1], gear_end_rpms[-1], int(2.5 * sr))
    profile.append(final_accel)

    # redline hold
    profile.append(numpy.full(int(redline_hold_time * sr), gear_end_rpms[-1]))

    # deceleration after redline hold
    decel_after_redline = numpy.linspace(
        gear_end_rpms[-1], gear_start_rpms[-1], int(decel_time_per_gear * sr)
    )
    profile.append(decel_after_redline)

    # deceleration + downshift blips
    for g in reversed(range(1, num_gears)):
        start, end = gear_start_rpms[g], gear_end_rpms[g - 1]

        # pre-blip dip (normal downshift blip)
        dip_duration = downshift_durations[g - 1]
        t = numpy.linspace(0, 1, int(dip_duration * sr))
        blip = start + (end - start) * easing(t)
        profile.append(blip)

        # deceleration for next gear
        decel = numpy.linspace(
            end, gear_start_rpms[g - 1], int(decel_time_per_gear * sr)
        )
        profile.append(decel)

    # idle hold after stop
    profile.append(numpy.full(int(idle_time * sr), base_rpm))

    return numpy.concatenate(profile)


def generate_engine_sound() -> None:
    rpm_profile = build_rpm_profile()
    total_samples = len(rpm_profile)
    output = numpy.zeros((total_samples, 2), dtype=numpy.float32)

    processing_pipeline = [
        # (
        #     bass_boost,
        #     {
        #         "gain_db": 10.0,
        #         "cutoff": 120.0,
        #         "fs": sr,
        #     },
        # ),
        (
            compressor,
            {
                "threshold_db": 0.0,
                "ratio": 8.0,
                "knee_db": 0.0,
                "makeup_gain_db": 0.0,
            },
        ),
    ]

    for cyl_idx, deg_offset in enumerate(firing_degrees):
        t = (deg_offset / crank_degrees_per_cycle) * (60.0 / base_rpm)

        while True:
            sample_index = int(t * sr)
            if sample_index >= total_samples:
                break

            rpm = rpm_profile[min(sample_index, total_samples - 1)]
            raw_pulse = get_pulse(rpm)

            # Apply the effects pipeline
            pulse_audio = apply_audio_effects_pipeline(raw_pulse, processing_pipeline)

            pan = cyl_idx / max(1, num_cylinders - 1)
            stereo_pulse = (pulse_audio * (1.0 - pan), pulse_audio * pan)
            stacked = numpy.stack(stereo_pulse, axis=-1)

            end_index = sample_index + len(pulse_audio)
            if end_index > total_samples:
                stacked = stacked[: total_samples - sample_index]

            output[sample_index : sample_index + len(stacked)] += stacked

            # advance by 720° (2 revolutions per firing)
            t += 60.0 / (rpm * (crank_degrees_per_cycle / 720.0))

    max_val = numpy.max(numpy.abs(output))
    if max_val > 0:
        output /= max_val  # normalize to prevent clipping

    soundfile.write(output_path, output, sr)
    print(f"Generated engine sound to: {output_path}")


if __name__ == "__main__":
    generate_engine_sound()
