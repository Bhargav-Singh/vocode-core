import time
import queue
from math import log10
from typing import List, Tuple

import numpy as np
import sounddevice as sd


def list_input_devices() -> List[Tuple[int, dict]]:
    devices = sd.query_devices()
    indexed = []
    for global_idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            indexed.append((global_idx, d))
    return indexed


def monitor_microphone(device_index: int, samplerate: int | None = None, blocksize: int = 2048):
    q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(f"[mic:{device_index}] status: {status}")
        # copy to avoid re-use by sounddevice
        q.put(indata.copy())

    dev_info = sd.query_devices(device=device_index)
    sr = int(samplerate or dev_info.get("default_samplerate", 44100))
    print(f"Opening input device {device_index}: {dev_info['name']} @ {sr} Hz")

    last_print = 0.0
    with sd.InputStream(
        device=device_index,
        channels=1,
        samplerate=sr,
        blocksize=blocksize,
        dtype="int16",
        callback=callback,
    ):
        print("Monitoring… press Ctrl+C to switch or 'q'+Enter to quit")
        try:
            while True:
                try:
                    chunk = q.get(timeout=0.5)
                except queue.Empty:
                    print("[mic] no frames yet…")
                    continue
                now = time.time()
                if now - last_print > 0.5:
                    samples = chunk.astype(np.int16)
                    if samples.size:
                        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
                        if rms > 0:
                            dbfs = 20.0 * log10(rms / 32768.0)
                            print(f"[mic] level: {dbfs:.1f} dBFS")
                            if dbfs > -3:
                                print("[mic] WARNING: close to clipping")
                        else:
                            print("[mic] silence")
                    last_print = now
        except KeyboardInterrupt:
            pass


def main():
    while True:
        inputs = list_input_devices()
        if not inputs:
            print("No input devices found.")
            return
        print("Available input devices:")
        for gidx, d in inputs:
            print(f"  {gidx}: {d['name']}")
        raw = input("Select input device index (or 'q' to quit): ").strip()
        if raw.lower() == "q":
            break
        try:
            idx = int(raw)
        except ValueError:
            print("Please enter a valid integer index.")
            continue
        if idx not in {g for g, _ in inputs}:
            print("Index not in input devices list.")
            continue
        monitor_microphone(idx)


if __name__ == "__main__":
    main()
