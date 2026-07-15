# -*- coding: utf-8 -*-
"""
事前学習用データ取得プログラムです。
This script collects EMG data for pre-training.

スポーツセンシング社製のDSPワイヤレス筋電位センサ（湿式）ハイエンドモデル
（SS-EMGW-HM）での動作確認をしています。
This script was tested with the Sports Sensing DSP wireless EMG sensor
(wet electrode, high-end model, SS-EMGW-HM).

また、このセンサを使う場合は、スポーツセンシング社のソフトウェア開発キット（SDK）が必要となります。
To use this sensor, the Sports Sensing software development kit (SDK)
is required.

SDKの入手条件や開発サポートについては、スポーツセンシング社の公式情報を確認してください。
Please check the official Sports Sensing information for SDK availability
and development support options.

Official SDK page / 公式SDKページ:  
https://www.sports-sensing.com/brands/labss/labss_software/sdk/wirelesssdk.html

この公開版では、DSPワイヤレスセンサ固有の通信部分は消してあります。
In this public version, the communication code specific to the DSP wireless sensor has been removed.

実際に動かす場合は、適切な通信コードを入れて下さい。
To run this script, please add the appropriate communication code for your own environment.

動作は、握り、反り、動作なしの順に行うように設定されています。
The movements are recorded in the following order: grip, wrist extension, and no movement.

find_peaksの上限と下限は、私たちと異なることが想定されます。
The upper and lower thresholds for peak detection may differ depending on your sensor,
subject, and experimental environment.

調整して実行してください。
Please adjust these parameters before running the script.
"""

import datetime
import logging
import threading
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import emg_data_collection as dc

# ログの設定
# Configure logging.
logging.basicConfig(level=logging.ERROR, filename='error.log')

def handle_error(e):
    logging.error(f"An error occurred / エラーが発生しました: {e}", exc_info=True)


def peak_plot(x, y, idx):
    """
    筋電位データをプロットし、検出されたピークを赤点で表示します。
    Plot EMG data and mark detected peaks with red dots.

    Args:
        x (array-like): x軸の値。通常はサンプル番号。
            Values for the x-axis, usually sample indices.
        y (array-like): 取得した筋電位データ。
            Recorded EMG signal.
        idx (array-like): ピークとして検出されたインデックス。
            Indices detected as peaks.

    Returns:
        None
    """
    plt.figure()
    plt.plot(x, y)
    plt.plot(x[idx], y[idx], "ro")
    plt.show()


def hampel(x, k, thr=3):
    """
    Hampelフィルタを用いて外れ値を補正します。
    Correct outliers using a Hampel filter.

    各点の近傍における中央値とMADに基づいて外れ値を検出し、
    外れ値を近傍の中央値で置き換えます。
    Outliers are detected based on the local median and MAD
    and replaced with the local median.

    Args:
        x (np.ndarray): 取得した筋電位データ。
            Recorded EMG signal.
        k (int): 各点の前後で参照する近傍範囲。
            Window radius around each sample.
        thr (float): 外れ値判定に用いる閾値。
            Threshold for outlier detection.

    Returns:
        tuple:
            output_x (np.ndarray): 外れ値補正後のデータ。
                Filtered signal.
            output_idx (np.ndarray): 外れ値として検出された位置。
                Indices marked as outliers.
    """
    array_size = len(x)
    idx = np.arange(array_size)
    output_x = x.copy()
    output_idx = np.zeros_like(x)
 
    for i in range(array_size):
        mask1 = np.where( idx >= (idx[i] - k) ,True, False)
        mask2 = np.where( idx <= (idx[i] + k) ,True, False)
        kernel = np.logical_and(mask1, mask2)
        median = np.median(x[kernel])
        std = 1.4826 * np.median(np.abs(x[kernel] - median))
        if np.abs(x[i] - median) > thr * std:
            output_idx[i] = 1
            output_x[i] = median
 
    # return output_x, output_idx.astype(bool)
    return output_x, output_idx

# 各動作で取得するサンプル数。
# Number of samples collected for each movement.
#
# 実験では余分にデータをとるため13サンプル取得していますが、5サンプルの使用です。
# In the experiment, 13 samples are collected to include extra data,
# but only 5 samples are used.
#
# 例:
# Example:
#   13 movements -> 4100 samples
#
# この値を変更する場合は、動作タイミングや動作なしデータの長さも調整してください。
# If this value is changed, also adjust the movement timing and
# the length of the no-movement data.

INITIAL_GRIP_EXCLUSION_SAMPLES = 451
INITIAL_WRIST_EXTENSION_EXCLUSION_SAMPLES = 451
INITIAL_NO_MOVEMENT_EXCLUSION_SAMPLES = 601
NO_MOVEMENT_REPEAT_COUNT = 8

SAMPLES_PER_MOVEMENT = 4100
retry_grip = "any key"
retry_wrist_extension = "any key"
retry_no_movement = "any key"

input("事前学習データ取得を開始します。Enterキーを押してください。 / Press Enter to start collecting pre-training data.")

dc.initialize_audio()

# シリアルポートを開いて通信準備を行います。
# Open the serial port and prepare communication.
#
# TODO:
# ここで使用するセンサSDKに合わせて Ser を初期化してください。
# Initialize `Ser` here according to your sensor SDK.
#
# Example:
# Ser = ...

Ser = None

try:
    # 握り動作の計測
    # Record grip movement.
    while retry_grip != "":

    # タイマー関数をバックグラウンドで実行し、動作の合図とデータ取得を同時に行います。
    # Run the timer function in the background so that movement cues and data collection occur simultaneously.
        beep_thread = threading.Thread(
            target=dc.notify_movement_timing_calibration,
            args=("grip",)
        )
        beep_thread.start()
        try:
            grip_data = dc.get_data(Ser, SAMPLES_PER_MOVEMENT)
        except Exception as e:
            print(e)
            traceback.print_exc()
            raise
        beep_thread.join()
        
        print("すべての処理が完了しました。 / All tasks completed.")
    
        grip_data = np.array(grip_data)
        grip_filtered, _ = hampel(grip_data, 10, thr=2)
        grip_calibration_max = np.max(grip_filtered)
        grip_time = np.arange(len(grip_filtered))
        
        # ピーク検出の閾値は、私たちの実験環境に合わせた設定です。
        # These peak-detection thresholds were tuned for our experimental environment. 
        grip_peaks = dc.peak_detect(
            grip_filtered,
            grip_calibration_max * 1 / 3,
            grip_calibration_max + 0.05,
            distance=200
        )
        grip_peaks = np.array(grip_peaks)
        
        # 計測開始直後は信号が安定しないため、ピーク候補から除外します。
        # Exclude early peaks because the signal may be unstable immediately after measurement starts.
        grip_peaks_to_delete = np.where(
            grip_peaks < INITIAL_GRIP_EXCLUSION_SAMPLES
        )
        grip_peaks = np.delete(grip_peaks, grip_peaks_to_delete)
        
        print("ピークの値 / Peak indices:", grip_peaks)
        
        peak_plot(grip_time, grip_filtered, grip_peaks)
        
        print("\n今、計測した筋電位データの図です。 / Plot of the recorded EMG data.\n")
        
        retry_grip = input(
            "やり直す場合はEnter以外のキーを入力してください。 / "
            "Press any key other than Enter to retry.\n"
        )
        
    
    grip_data_all = grip_filtered.copy()
    
    print("\n 反りデータを計測します。 / Start recording wrist extension data.\n")
    
    # 手の反り動作の計測
    # Record wrist extension movement.
    while retry_wrist_extension != "":
        
        beep_thread = threading.Thread(
            target=dc.notify_movement_timing_calibration,
            args=("extend_wrist",)
        )
        beep_thread.start()
        try:
            wrist_extension_data = dc.get_data(Ser, SAMPLES_PER_MOVEMENT)
        except Exception as e:
            print(e)
            traceback.print_exc()
            raise
        beep_thread.join()
        
        print("すべての処理が完了しました。 / All tasks completed.")
        
        wrist_extension_data = np.array(wrist_extension_data)
        wrist_extension_filtered, _ = hampel(wrist_extension_data, 10, thr=2)
        wrist_extension_calibration_max = np.max(wrist_extension_filtered)
        wrist_extension_time = np.arange(len(wrist_extension_filtered))
        
        # ピーク検出の閾値は、私たちの実験環境に合わせた設定です。
        # These peak-detection thresholds were tuned for our experimental environment.
        
        wrist_extension_peaks = dc.peak_detect(
            wrist_extension_filtered,
            wrist_extension_calibration_max * 1 / 3,
            wrist_extension_calibration_max + 0.05,
            distance=200
        )
        
        wrist_extension_peaks = np.array(wrist_extension_peaks)
        wrist_extension_peaks_to_delete = np.where(
            wrist_extension_peaks < INITIAL_WRIST_EXTENSION_EXCLUSION_SAMPLES
        )
        wrist_extension_peaks = np.delete(wrist_extension_peaks, wrist_extension_peaks_to_delete)
        
        print("\n今、計測した筋電位データの図です。 / Plot of the recorded EMG data.\n")
            
        peak_plot(wrist_extension_time, wrist_extension_filtered, wrist_extension_peaks)
        retry_wrist_extension = input(
            "やり直す場合はEnter以外のキーを入力してください。 / "
            "Press any key other than Enter to retry.\n"
        )
        
    
    wrist_extension_data_all = wrist_extension_filtered.copy()
    
    # 動作なし状態の計測
    # Record the no-movement condition.
    while retry_no_movement != "":
    
        beep_thread = threading.Thread(target=dc.notify_no_movement)
        beep_thread.start()
        try:
            no_movement_data = dc.get_no_movement_data(Ser)
        except Exception as e:
            print(e)
            traceback.print_exc()
            raise
        beep_thread.join()
            
        print("すべての処理が完了しました。 / All tasks completed.")
        
        no_movement_data = np.array(no_movement_data)
        no_movement_filtered, _ = hampel(no_movement_data, 10, thr=2)
        no_movement_time = np.arange(len(no_movement_filtered))
        no_movement_peaks = dc.peak_detect(no_movement_filtered, 0.001, 0.3, 200)
        no_movement_peaks = np.array(no_movement_peaks)
        no_movement_peaks_to_delete = np.where(
            no_movement_peaks < INITIAL_NO_MOVEMENT_EXCLUSION_SAMPLES
        )
        no_movement_peaks = np.delete(no_movement_peaks, no_movement_peaks_to_delete)
        
        peak_plot(no_movement_time, no_movement_filtered, no_movement_peaks)
        
        retry_no_movement = input(
            "やり直す場合はEnter以外のキーを入力してください。 / "
            "Press any key other than Enter to retry.\n"
        )
        
    no_movement_data_all = no_movement_filtered.copy()
    
    # 動作なしデータは比較的短時間で取得し、繰り返して学習データ長に合わせます。
    # The no-movement data is recorded for a shorter duration and repeated
    # to match the length of the training data.
    #
    # ここでは動作なしデータを8回繰り返し、必要なサンプル数だけ切り出します。
    # Here, the no-movement data is repeated 8 times and trimmed to the required number of samples.
    no_movement_data_all = np.tile(no_movement_data_all, NO_MOVEMENT_REPEAT_COUNT)[50:50 + SAMPLES_PER_MOVEMENT]
    
    input()
    
    
    # ポートを閉じる
    # Close the port

    x_train_all = np.concatenate([[grip_data_all],[wrist_extension_data_all],[no_movement_data_all]])
    
    current_date = datetime.date.today().strftime('%Y%m%d')
    
    base_dir = Path.home() / "Desktop" / "EMG_data"
    today_dir = base_dir / current_date
    x_train_dir = today_dir / "x_train_data"
    x_train_dir.mkdir(parents=True, exist_ok=True)

    # 保存先のファイルパス
    # Output file path.
    file_path = x_train_dir / "x_train_all"
    np.save(file_path, x_train_all)

    file_peak_path_grip_data = x_train_dir / "peak_data_grip_data.npy"
    file_peak_path_wrist_extension_data = x_train_dir / "peak_data_wrist_extension_data.npy"
    
    # ピークデータをファイルに保存します。
    # Save the peak data to files.
    np.save(file_peak_path_grip_data, grip_peaks)
    np.save(file_peak_path_wrist_extension_data, wrist_extension_peaks)
    
    print(
    f"データが {file_peak_path_grip_data} と {file_peak_path_wrist_extension_data} に保存されました。 / "
    f"Data saved to {file_peak_path_grip_data} and {file_peak_path_wrist_extension_data}."
    )

except KeyboardInterrupt:
    print("キーボード割り込みが発生しました。 / Keyboard interrupt detected.")
    
except Exception as e:
    print(f"エラーが発生しました / An error occurred: {e}")
    handle_error(e)
    traceback.print_exc()
finally:
    # センサSDKに応じてポートを閉じてください。
    # Close the port according to your sensor SDK.
    pass
