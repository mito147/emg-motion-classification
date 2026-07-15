# EMG Pre-training Data Collection

This repository provides scripts for collecting EMG data for pre-training.

事前学習用の筋電位データ取得プログラムです。  

This repository implements the EMG data collection program related to Section 2.3.1 of the following paper:

本リポジトリは、以下の論文の 2.3.1 節に関連する筋電位データ取得プログラムを実装したものです。

> Masataka Ito and Takuya Kitamoto,  
> “On Real-Time Motion Classification by Machine Learning Using Myoelectric Potential,”  
> *Bulletin of the Faculty of Education, Yamaguchi University*, Vol. 74, pp. 163–171, 2025.  
>
> 伊藤正剛・北本卓也,  
> 「筋電位での機械学習によるリアルタイム動作識別について」,  
> 山口大学教育学部研究論叢, 第74巻, pp. 163–171, 2025.  
>
> Paper URL: https://petit.lib.yamaguchi-u.ac.jp/30053

This script was tested with the Sports Sensing DSP wireless EMG sensor  
(wet electrode, high-end model, SS-EMGW-HM).

スポーツセンシング社製のDSPワイヤレス筋電位センサ（湿式）ハイエンドモデル  
（SS-EMGW-HM）で動作確認を行いました。

## Overview / 概要

This program records EMG data for three conditions:

- Grip
- Wrist extension
- No movement

このプログラムでは、以下の3条件の筋電位データを取得します。

- 握り
- 反り
- 動作なし

The collected data is intended to be used as pre-training data for machine learning.

取得したデータは、機械学習における事前学習用データとして使用することを想定しています。

## Implementation Scope / 実装範囲

This public version focuses on the data acquisition and preprocessing part of the workflow.  
It includes movement timing cues, EMG data acquisition placeholders, Hampel filtering, peak detection, plotting, and saving of pre-training data.

この公開版では、ワークフローのうちデータ取得および前処理に関する部分を中心に実装しています。  
動作タイミングの通知、筋電位データ取得部分のプレースホルダ、Hampelフィルタ、ピーク検出、プロット、事前学習用データの保存を含みます。

## Important Note / 重要な注意

Sensor-specific communication code has been removed from this public version.

この公開版では、センサSDK固有の通信処理を削除しています。

Therefore, this script will raise `NotImplementedError` unless you implement the following functions according to your own sensor SDK:

そのため、使用するセンサSDKに合わせて以下の関数を実装しない限り、このスクリプトは `NotImplementedError` を出して停止します。

```python
get_data()
get_no_movement_data()
```

## Sports Sensing SDK / スポーツセンシングSDK

To communicate with the Sports Sensing DSP wireless EMG sensor, the Sports Sensing software development kit (SDK) is required.

スポーツセンシング社製DSPワイヤレス筋電位センサと通信するには、スポーツセンシング社のソフトウェア開発キット（SDK）が必要です。

The SDK-specific communication code is not included in this repository.

本リポジトリには、スポーツセンシング社SDK固有の通信処理は含まれていません。

Please check the official Sports Sensing website for SDK availability, terms of use, and support options.

SDKの入手方法、利用条件、サポート内容については、スポーツセンシング社の公式Webサイトを確認してください。

Official SDK page / 公式SDKページ:  
https://www.sports-sensing.com/brands/labss/labss_software/sdk/wirelesssdk.html

## Files / ファイル構成

```text
.
├── collect_pretraining_emg_data.py
├── emg_data_collection.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

### `collect_pretraining_emg_data.py`

Main script for collecting EMG data.

筋電位データを取得するメインスクリプトです。

The main experimental parameters are defined as constants in this file.

主な実験条件は、このファイル内の定数として定義しています。

### `emg_data_collection.py`

Helper module for EMG data collection.

筋電位データ取得用の補助モジュールです。

This module provides helper functions for:

- notifying movement timing
- playing audio cues
- detecting peaks from ARV data
- defining placeholders for sensor SDK-based data acquisition

このモジュールでは、以下の補助機能を提供します。

- 動作タイミングの通知
- 音声キューの再生
- ARVデータからのピーク検出
- センサSDKに応じたデータ取得部分のプレースホルダ

Audio cues are played using `.wav` files placed in the `audio` folder on the Windows Desktop.

音声キューは、Windowsのデスクトップ上にある `audio` フォルダ内の `.wav` ファイルを使用して再生します。

The module uses `pygame` for audio playback.  
If beep sounds are used, it relies on the Windows-only `winsound` module.

このモジュールでは、音声再生に `pygame` を使用します。  
ビープ音を使用する場合は、Windows専用の `winsound` モジュールを使用します。

The functions `get_data()` and `get_no_movement_data()` are not implemented in this public version.  
Please implement them according to your own sensor SDK.

この公開版では、`get_data()` と `get_no_movement_data()` は実装していません。  
使用するセンサSDKに合わせて、これらの関数を実装してください。

## Requirements / 必要な環境

This program is intended to run on Windows.

このプログラムはWindows環境での使用を想定しています。

Python 3.11 is recommended.

Python 3.11 の使用を推奨します。

Python packages are listed in `requirements.txt`.

必要なPythonパッケージは `requirements.txt` に記載しています。

## Audio Files / 音声ファイル

This program assumes that audio cue files are placed in the `audio` folder on the Windows Desktop.

このプログラムでは、Windowsのデスクトップ上にある `audio` フォルダ内の音声ファイルを使用する想定です。

Expected file paths:

想定しているファイルパスは以下の通りです。

```text
Desktop/audio/nigiri.wav
Desktop/audio/sori.wav
Desktop/audio/finish.wav
```

These audio files are used for movement timing cues.

これらの音声ファイルは、動作タイミングを知らせるために使用します。

## Parameters / パラメータ

Main experimental parameters are defined as constants in `collect_pretraining_emg_data.py`.

主な実験条件は `collect_pretraining_emg_data.py` 内の定数として定義しています。

```python
INITIAL_GRIP_EXCLUSION_SAMPLES = 451
INITIAL_WRIST_EXTENSION_EXCLUSION_SAMPLES = 451
INITIAL_NO_MOVEMENT_EXCLUSION_SAMPLES = 601
NO_MOVEMENT_REPEAT_COUNT = 8
SAMPLES_PER_MOVEMENT = 4100
```

Peak detection thresholds were tuned for our experimental environment.

ピーク検出のしきい値は、私たちの実験環境に合わせて調整した値です。

Please adjust these parameters depending on your sensor, subject, and experimental environment.

使用するセンサ、被験者、実験環境に応じて、これらの値を調整してください。

## Output / 出力

The collected data is saved under the following directory:

取得したデータは、以下のディレクトリに保存されます。

```text
Desktop/EMG_data/YYYYMMDD/x_train_data/
```

Example output files:

出力されるファイルの例です。

```text
x_train_all.npy
peak_data_grip_data.npy
peak_data_wrist_extension_data.npy
```

Generated data files such as `.npy`, `.csv`, `.png`, and log files should not usually be uploaded to GitHub.

`.npy`, `.csv`, `.png` などの生成データやログファイルは、通常GitHubにはアップロードしないでください。

## Authors / 著者

This repository was prepared by Masataka Ito and Takuya Kitamoto.

本リポジトリは、伊藤正剛および北本卓也により作成されました。

## License / ライセンス

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

このプロジェクトはMITライセンスのもとで公開されています。詳細は [LICENSE](LICENSE) ファイルを確認してください。

## Disclaimer / 免責事項

This software is provided for research and educational purposes.

本ソフトウェアは、研究および教育目的で提供されています。

The author is not responsible for any issues, damages, or data loss caused by the use of this software.

本ソフトウェアの使用によって生じたいかなる問題、損害、データ損失についても、作者は責任を負いません。
