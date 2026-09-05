# -*- coding: utf-8 -*-
"""

全ての筋電位データから切り出しを行います。
Extract fixed-width windows from the collected EMG data.

ピーク数（サンプル数）とピーク片側のデータ数を指定して訓練データを作成します。
Create training data by specifying the number of peaks (samples) and the
number of values extracted on each side of a peak.

"""

import datetime
from pathlib import Path
import sys

import numpy as np

def extract_peak_windows(
    peak_indices,
    emg_signal,
    sample_count,
    half_window,
):
    try:
        windows = np.empty(
            (0, 2 * half_window)
        )

        for i in range(sample_count):
            peak = peak_indices[i]
            peak_window = emg_signal[
                peak - half_window:
                peak + half_window
            ]
            windows = np.vstack([
                windows,
                peak_window,
            ])

        return windows

    except (IndexError, ValueError) as error:
        raise ValueError(
            "ピーク数が不足しているか、ピーク前後のデータを"
            "必要な長さで切り出せませんでした。 / "
            "Could not extract peak-centered windows."
        ) from error

#動作なしのデータをサンプル数分だけ保存する。
def extract_no_movement_windows(
    emg_signal,
    sample_count,
    half_window,
):
    """
    動作なし信号のオフセット200以降から固定長データを切り出します。
    Extract fixed-width windows from the no-movement signal, beginning at
    offset 200. Peak indices are not required for the flat signal.

    Args:
        emg_signal (array-like): 動作なし状態の筋電位データ（ARV）。
            No-movement EMG data (ARV).
        sample_count (int): 切り出すサンプル数。
            Number of windows to extract.
        half_window (int): 各サンプルの半分のデータ数。
            Half the number of values in each window.

    Returns:
        np.ndarray: 形状（sample_count, 2 * half_window）の切り出しデータ。
            Extracted windows with shape (sample_count, 2 * half_window).
    """
    windows = np.empty((0, 2 * half_window))
    for i in range(sample_count):
        no_movement_window = emg_signal[i+200:i+200+2*half_window]
        windows = np.vstack([windows,no_movement_window])
    
    return windows

current_date = datetime.date.today().strftime(
    "%Y%m%d"
)

folder_path = (
    Path.home()
    / "Desktop"
    / "EMG_data"
    / current_date
    / "x_train_data"
)

input_dir = folder_path

if not input_dir.is_dir():
    raise FileNotFoundError(
        f"入力フォルダが見つかりません: {input_dir} / "
        f"Input directory was not found: {input_dir}"
    )

CLASS_COUNT = 3

X_TRAIN_ALL_FILENAME = "x_train_all.npy"
GRIP_PEAKS_FILENAME = "peak_data_grip_data.npy"
WRIST_EXTENSION_PEAKS_FILENAME = "peak_data_wrist_extension_data.npy"

x_train_all = np.load(
    input_dir / X_TRAIN_ALL_FILENAME,
    allow_pickle=False,
)
peak_data_grip = np.load(
    input_dir / GRIP_PEAKS_FILENAME,
    allow_pickle=False,
)
peak_data_wrist_extension = np.load(
    input_dir / WRIST_EXTENSION_PEAKS_FILENAME,
    allow_pickle=False,
)

while True:
    try:        
        # 行数とピーク片側のデータ数を入力します。
        # Enter the sample count and the number of values on each side of a peak.
        sample_count, half_window = map(int, input(
            "サンプル数とピーク片側のデータ数を、スペースで区切って入力してください。 / "
            "Enter sample_count and half_window separated by a space:\n"
        ).split())
        
        if sample_count <= 0 or half_window <= 0:
            print(
                "サンプル数とピーク片側のデータ数には、1以上の整数を入力してください。 / "
                "Enter positive integers for sample_count and half_window.\n"
            )
            continue
     
        print(
            "標準値は sample_count が5または10、half_window が10または17です。 / "
            "Standard values are 5 or 10 for sample_count and 10 or 17 for half_window.\n"
        )
        
        # 入力値を確認します。 / Confirm the entered values.
        confirm = input(
            f"{sample_count}行 × {2 * half_window}列に設定されました。"
            "この設定でよろしいですか？ (y/n) / "
            f"Configured shape: {sample_count} rows × {2 * half_window} columns. "
            "Continue? (y/n):\n"
        )

        # 設定を確定します。 / Confirm the configuration.
        if confirm.lower() == 'y':
            print("\n設定が確定されました。 / Configuration confirmed.\n")
            break
        elif confirm.lower() == 'n':
            print("設定をやり直してください。 / Please enter the settings again.\n")
        else:
            print(
                "無効な入力です。'y' または 'n' を入力してください。 / "
                "Invalid input. Enter 'y' or 'n'.\n"
            )
    except ValueError:
        print(
            "無効な入力です。2つの整数をスペースで区切って入力してください。 / "
            "Invalid input. Enter two integers separated by a space.\n"
        )

# 3クラスのデータを結合して訓練データを作ります。
# Combine the three classes into the training array.
grip_windows = extract_peak_windows(
    peak_data_grip,
    x_train_all[0],
    sample_count,
    half_window,
)

wrist_extension_windows = extract_peak_windows(
    peak_data_wrist_extension,
    x_train_all[1],
    sample_count,
    half_window,
)

no_movement_windows = extract_no_movement_windows(
    x_train_all[2],
    sample_count,
    half_window,
)

#データを結合させ訓練ファイルを作る。
x_train = np.concatenate([
    grip_windows,
    wrist_extension_windows,
    no_movement_windows,
])
x_train = x_train.reshape(CLASS_COUNT*sample_count,2*half_window)

x_train = x_train.astype(
    np.float32,
    copy=False,
)

grip_end = sample_count
wrist_end = 2 * sample_count
no_movement_end = 3 * sample_count

grip_mean = np.mean(x_train[:grip_end])
wrist_extension_mean = np.mean(x_train[grip_end:wrist_end])
no_movement_mean = np.mean(x_train[wrist_end:no_movement_end])

warning_flag = False

if grip_mean != 0 and wrist_extension_mean / grip_mean > 0.7:
    print(
        "【警告】握りと反りの大きさが近すぎます。 / "
        "Warning: The grip and wrist-extension signal levels are too similar."
    )
    warning_flag = True

if wrist_extension_mean != 0 and no_movement_mean / wrist_extension_mean > 0.7:
    print(
        "【警告】動作なしと反りの大きさが近すぎます。 / "
        "Warning: The no-movement and wrist-extension signal levels are too similar."
    )
    warning_flag = True

# 警告が出た場合のみ、保存するかどうかを人間に確認する
if warning_flag:
    force_save = input("警告が出ています。それでもデータを保存しますか？ (y/n):\n")
    if force_save.lower() != 'y':
        print("保存を中止しました。 / Save cancelled.")
        sys.exit(0)

# ファイル名に3クラス分の総サンプル数と特徴量数を含めます。
# Include the total sample count for three classes and the feature count.
file_path = folder_path / (
    f"x_train_{CLASS_COUNT * sample_count}"
    f"_2x{half_window}.npy"
)

np.save(
    file_path,
    x_train,
    allow_pickle=False,
)

print(
    f"訓練データを保存しました: {file_path} / "
    f"Training data saved: {file_path}"
)



