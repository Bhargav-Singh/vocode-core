import math
from typing import List, Tuple

import numpy as np
import sounddevice as sd


def list_output_devices() -> List[Tuple[int, dict]]:
    devices = sd.query_devices()
    indexed = []
    for global_idx, d in enumerate(devices):
        if d.get("max_output_channels", 0) > 0:
            indexed.append((global_idx, d))
    return indexed


def play_test_tone(device_index: int, samplerate: int | None = None, freq: float = 440.0, level_dbfs: float = -20.0):
    dev_info = sd.query_devices(device=device_index)
    sr = int(samplerate or dev_info.get("default_samplerate", 44100))
    print(f"Opening output device {device_index}: {dev_info['name']} @ {sr} Hz")

    # Convert level dBFS (for float32, full-scale=1.0) to linear amplitude
    amplitude = 10 ** (level_dbfs / 20.0)
    phase = 0.0
    two_pi = 2 * math.pi

    def callback(outdata, frames, time_info, status):  # noqa: ARG001
        nonlocal phase
        if status:
            print(f"[spk:{device_index}] status: {status}")
        t = (np.arange(frames) + phase) / sr
        tone = amplitude * np.sin(two_pi * freq * t)
        outdata[:, 0] = tone.astype(np.float32)
        # advance phase to keep continuity
        phase = (phase + frames) % sr

    print("Playing 440 Hz test tone at ~-20 dBFS. Press Ctrl+C to stop.")
    with sd.OutputStream(
        device=device_index,
        channels=1,
        samplerate=sr,
        dtype="float32",
        callback=callback,
        blocksize=1024,
    ):
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            pass


def main():
    while True:
        outputs = list_output_devices()
        if not outputs:
            print("No output devices found.")
            return
        print("Available output devices:")
        for gidx, d in outputs:
            print(f"  {gidx}: {d['name']}")
        raw = input("Select output device index (or 'q' to quit): ").strip()
        if raw.lower() == "q":
            break
        try:
            idx = int(raw)
        except ValueError:
            print("Please enter a valid integer index.")
            continue
        if idx not in {g for g, _ in outputs}:
            print("Index not in output devices list.")
            continue
        play_test_tone(idx)


if __name__ == "__main__":
    main()
