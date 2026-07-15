# -*- coding: utf-8 -*-
"""
筋電位データ取得用の補助モジュールです。
EMG data collection helper module.

このモジュールでは、以下の処理を行います。
This module provides helper functions for:
- 動作タイミングの通知 / notifying movement timing
- 音声キューの再生（Windows環境を想定）
  playing audio cues, assuming a Windows environment
- センサSDKを用いたデータ取得 / collecting data through a sensor SDK
- ARVデータからのピーク検出 / detecting peaks from ARV data

このプログラムは、Windows環境での使用を想定しています。
This program is intended to run on Windows.

音声キューの再生には、Windowsのデスクトップ上にある audio フォルダ内の
.wav ファイルを使用します。
Audio cues are played from .wav files located in the audio folder
on the Windows Desktop.

また、ビープ音を使用する場合は Windows専用の winsound モジュールを使用します。
If beep sounds are used, this program relies on the Windows-only
winsound module.

センサ固有の通信コードは削除しています。
The sensor-specific communication code has been removed.

実際に使用する場合は、使用するSDKに合わせて get_data() と
get_no_movement_data() を実装してください。
Please implement get_data() and get_no_movement_data() according to your sensor SDK.
"""

import threading
import time
from pathlib import Path

import pygame
from scipy import signal

try:
    import winsound
except ImportError:
    winsound = None

AUDIO_DIR = Path.home() / "Desktop" / "audio"

GRIP_SOUND_PATH = AUDIO_DIR / "nigiri.wav"
WRIST_EXTENSION_SOUND_PATH = AUDIO_DIR / "sori.wav"
FINISH_SOUND_PATH = AUDIO_DIR / "finish.wav"


def initialize_audio():
    """
    pygameのmixerを初期化します。
    Initialize the pygame mixer.
    """
    pygame.mixer.init()


def play_sound_in_thread(sound_path):
    """
    音声ファイルを別スレッドで再生します。
    Play a sound file in a separate thread.

    Args:
        sound_path (str | Path): 再生する音声ファイルのパス。
            Path to the sound file.
    """
    def play_sound():
        pygame.mixer.music.load(str(sound_path))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    threading.Thread(target=play_sound, daemon=True).start()


def notify_movement_timing_calibration(
    movement,
    interval=3.2,
    use_beep=False,
    initial_delay=7,
    repeat=13,
):
    """
    動作を行うタイミングを文字と音で知らせる関数です。
    Notify the participant of movement timing using text and sound.

    Args:
        movement (str): 動作ID。
            Movement ID. Use "grip" or "extend_wrist".
        interval (float): 音と音の間隔。秒単位。
            Interval between cues in seconds.
        use_beep (bool): Trueの場合はビープ音を使用します。
            If True, use a beep sound instead of audio files.
        initial_delay (float): 開始前の待ち時間。秒単位。
            Delay before the first cue in seconds.
        repeat (int): 繰り返し回数。
            Number of repetitions.
    """
    note_freq = 540
    note_duration = 800

    movement_info = {
        "grip": {
            "japanese_instruction": "手を握ってください",
            "english_instruction": "please grip",
            "sound_path": GRIP_SOUND_PATH,
        },
        "extend_wrist": {
            "japanese_instruction": "手を反ってください",
            "english_instruction": "please extend your wrist",
            "sound_path": WRIST_EXTENSION_SOUND_PATH,
        },
    }

    time.sleep(initial_delay)

    for i in range(1, repeat + 1):
        info = movement_info.get(movement)

        print("データを取得します。 / Collecting data.")

        if info is None:
            print(f"Trial {i}: {movement} / {i}回目: {movement}")
            sound_path = None
        else:
            print(
                f"{i}回目、{info['japanese_instruction']}。 / "
                f"Trial {i}: {info['english_instruction']}."
            )
            sound_path = info["sound_path"]

        if use_beep:
            if winsound is None:
                raise RuntimeError(
                    "winsound is only available on Windows. / "
                    "winsound は Windows でのみ使用できます。"
                )
            winsound.Beep(note_freq, note_duration)
        elif sound_path is not None:
            play_sound_in_thread(sound_path)

        time.sleep(interval)
        

def notify_no_movement(interval=2.0, initial_delay=7, repeat=3):
    """
    動作なし状態を記録するタイミングを文字で知らせます。
    Notify the participant of no-movement recording timing.
    """
    time.sleep(initial_delay)

    for i in range(1, repeat + 1):
        print("データを取得します。 / Collecting data.")
        print(
            f"{i}回目、動作なしデータ取得中です。力を抜いてください。 / "
            f"Trial {i}: recording no-movement data. Please relax."
        )
        time.sleep(interval)

    play_sound_in_thread(FINISH_SOUND_PATH)


def get_data(sensor, samples_per_movement):
    """
    SDKに合わせてEMGデータを取得します。
    Collect EMG data using your sensor SDK.

    Args:
        sensor: センサSDKの通信オブジェクト。
            Sensor communication object.
        samples_per_movement (int): 取得するサンプル数。
            Number of samples to collect.

    Returns:
        array-like: 取得したEMGデータ。
            Recorded EMG data.
    """
    raise NotImplementedError(
        "Please implement get_data() according to your sensor SDK. / "
        "使用するセンサSDKに合わせて get_data() を実装してください。"
    )


def get_no_movement_data(sensor):
    """
    SDKに合わせて動作なし状態のEMGデータを取得します。
    Collect no-movement EMG data using your sensor SDK.

    Args:
        sensor: センサSDKの通信オブジェクト。
            Sensor communication object.

    Returns:
        array-like: 動作なし状態のEMGデータ。
            Recorded no-movement EMG data.
    """
    raise NotImplementedError(
        "Please implement get_no_movement_data() according to your sensor SDK. / "
        "使用するセンサSDKに合わせて get_no_movement_data() を実装してください。"
    )


def peak_detect(arv_data, lower_limit, upper_limit, distance=150):
    """
    取得した1動作分の筋電位データARVからピークを検出します。
    Detect peaks from ARV data for one movement.

    Args:
        arv_data (array-like): 1動作分のARVデータ。
            ARV data for one movement.
        lower_limit (float): ピーク検出の下限値。
            Lower threshold for peak detection.
        upper_limit (float): ピーク検出の上限値。
            Upper threshold for peak detection.
        distance (int): ピーク間の最小距離。
            Minimum distance between peaks.

    Returns:
        np.ndarray: ピークのインデックス。
            Indices of detected peaks.
    """
    peaks = signal.find_peaks(
        arv_data,
        height=(lower_limit, upper_limit),
        distance=distance,
    )
    return peaks[0]

