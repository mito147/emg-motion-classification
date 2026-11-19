# EMG Motion Classification Reference Code

## リポジトリの位置づけ / Repository Status

本リポジトリは、EMGデータの収集、ピーク検出、学習データの作成、
MLP学習、リアルタイムのカーソル操作の流れを示すための参考実装およびサンプルコードです。

そのまま実行できる完成済みアプリケーションではありません。
公開版ではハードウェアおよびセンサSDK固有の通信処理を除外しています。
実際に使用する場合は、使用するセンサ、SDK、被験者、実験環境に合わせて、
関数やパラメータを調整してください。

This repository provides reference and sample code illustrating the
workflow for EMG data collection, peak detection, training-data
preparation, MLP training, and real-time cursor control.

It is not a standalone, ready-to-run application. Hardware- and
SDK-specific communication code has been omitted from the public
version. Some functions and parameters must be adapted to the user's
sensor, SDK, subject, and experimental environment.

## 実装範囲 / Implementation Scope

本リポジトリは、筋電位データ取得、特徴抽出、MLP学習、カーソル操作の参考実装です。以下の論文の2.3.1〜2.3.3節では、基礎となるデータ取得、ラベル設定、学習・推論の流れを説明しています。

This repository provides reference implementations for EMG data collection, feature extraction, MLP training, and cursor control. Sections 2.3.1–2.3.3 of the following paper describe the underlying acquisition, labeling, and training/inference workflow:

> 伊藤正剛・北本卓也,  
> 「筋電位での機械学習によるリアルタイム動作識別について」,  
> 山口大学教育学部研究論叢, 第74巻, pp. 163–171, 2025.  
>
> Masataka Ito and Takuya Kitamoto,  
> “On Real-Time Motion Classification by Machine Learning Using Myoelectric Potential,”  
> *Bulletin of the Faculty of Education, Yamaguchi University*, Vol. 74, pp. 163–171, 2025.  
>
> Paper URL: https://petit.lib.yamaguchi-u.ac.jp/30053

PyTorchによる学習・リアルタイム推論の背景については、次の論文も関連します。

The following paper also provides background on using PyTorch for training and real-time inference:

> 伊藤正剛・北本卓也,  
> 「筋電位でのジェスチャー操作におけるKerasとPyTorchの性能比較」,  
> 山口大学教育学部研究論叢, 第75巻, pp. 235–244, 2026.
>
> Masataka Ito and Takuya Kitamoto,  
> “Performance Comparison of Keras and PyTorch in Gesture Manipulation Using Myoelectric Potentials,”  
> *Bulletin of the Faculty of Education, Yamaguchi University*, Vol. 75, pp. 235–244, 2026.  

**本コードは公開用の参考実装であり、論文の全実験を再現するコード一式ではありません。** 学習条件は論文と異なる場合があります。第2段階の機械学習による識別は含まず、カーソル操作はMLPの予測系列に対するルールで判定します。

**This is a public reference implementation, not a complete reproduction of the papers' experiments.** Training settings may differ. The second-stage machine-learning classifier is not included; cursor actions use rules applied to MLP prediction sequences.

収集コードの説明には、スポーツセンシング社製のDSPワイヤレス筋電位センサ（湿式）ハイエンドモデル  
（SS-EMGW-HM）での動作確認が記載されています。

The collection code documentation reports testing with the Sports Sensing DSP wireless EMG sensor  
(wet electrode, high-end model, SS-EMGW-HM).

## 概要 / Overview

このプログラムでは、以下の3条件の筋電位データを取得します。

- 握り
- 反り
- 動作なし

This program records EMG data for three conditions:

- Grip
- Wrist extension
- No movement

取得したデータは、機械学習における事前学習用データとして使用することを想定しています。

The collected data is intended to be used as pre-training data for machine learning.

収集した事前学習用データに対して、特徴抽出プログラムとMLP学習プログラムを順に実行することで、学習済みモデルファイルを生成できます。

By running the feature extraction and MLP training programs sequentially on the collected pre-training data, you can generate a trained model file.

## 重要な注意 / Important Note

この公開版では、センサSDK固有の通信処理を削除しています。

Sensor-specific communication code has been removed from this public version.

そのため、使用するセンサSDKに合わせて以下の関数を実装しない限り、このスクリプトは `NotImplementedError` を出して停止します。

Therefore, this script will raise `NotImplementedError` unless you implement the following functions according to your own sensor SDK:

```python
get_data(sensor, samples_per_movement)
get_no_movement_data(sensor)
```

収集プログラムの `Ser` の初期化と、SDKに応じた終了処理も必要です。カーソル操作プログラムでは `Ser` と `ARV` が未定義のため、接続を初期化し、`input_buffer.append(ARV)` の前に新しいARV値を受信してください。未実装のままでは該当箇所で `NameError` になります。収集時と推論時で信号の単位・倍率・サンプリング条件をそろえてください。

Also initialize `Ser` in the collection program and implement SDK-specific shutdown. In the cursor-control program, `Ser` and `ARV` are undefined: initialize the connection and receive a new ARV value before `input_buffer.append(ARV)`. Otherwise, execution reaches a `NameError`. Use consistent signal units, scaling, and sampling conditions for collection and inference.

## スポーツセンシングSDK / Sports Sensing SDK

スポーツセンシング社製DSPワイヤレス筋電位センサと通信するには、スポーツセンシング社のソフトウェア開発キット（SDK）が必要です。

To communicate with the Sports Sensing DSP wireless EMG sensor, the Sports Sensing software development kit (SDK) is required.

本リポジトリには、スポーツセンシング社SDK固有の通信処理は含まれていません。

The SDK-specific communication code is not included in this repository.

SDKの入手方法、利用条件、サポート内容については、スポーツセンシング社の公式Webサイトを確認してください。

Please check the official Sports Sensing website for SDK availability, terms of use, and support options.

公式SDKページ / Official SDK page:  
https://www.sports-sensing.com/brands/labss/labss_software/sdk/wirelesssdk.html

## ファイル構成 / Files

```text
.
├── collect_pretraining_emg_data.py
├── emg_data_collection.py
├── prepare_emg_training_data.py
├── training_emg_mlp_pytorch.py
├── emg_motion_classification_cursor_control.py
└── README.md
```

### `collect_pretraining_emg_data.py`

筋電位データを取得するメインスクリプトです。

Main script for collecting EMG data.

主な実験条件は、このファイル内の定数として定義しています。

The main experimental parameters are defined as constants in this file.

### `emg_data_collection.py`

筋電位データ取得用の補助モジュールです。

Helper module for EMG data collection.

このモジュールでは、以下の補助機能を提供します。

- 動作タイミングの通知
- 音声キューの再生
- ARVデータからのピーク検出
- センサSDKに応じたデータ取得部分のプレースホルダ

This module provides helper functions for:

- notifying movement timing
- playing audio cues
- detecting peaks from ARV data
- defining placeholders for sensor SDK-based data acquisition

音声キューは、Windowsのデスクトップ上にある `audio` フォルダ内の `.wav` ファイルを使用して再生します。

Audio cues are played using `.wav` files placed in the `audio` folder on the Windows Desktop.

このモジュールでは、音声再生に `pygame` を使用します。  
ビープ音を使用する場合は、Windows専用の `winsound` モジュールを使用します。

The module uses `pygame` for audio playback.  
If beep sounds are used, it relies on the Windows-only `winsound` module.

この公開版では、`get_data()` と `get_no_movement_data()` は実装していません。  
使用するセンサSDKに合わせて、これらの関数を実装してください。

The functions `get_data()` and `get_no_movement_data()` are not implemented in this public version.  
Please implement them according to your own sensor SDK.

### `prepare_emg_training_data.py`

収集した連続EMG信号とピークインデックスから、MLP学習用の2次元NumPy配列を作成するスクリプトです。

This script converts the collected continuous EMG signals and peak indices into a two-dimensional NumPy array for MLP training.

既定では、`Desktop/EMG_data/YYYYMMDD/x_train_data/` から以下のファイルを読み込みます。

By default, it reads the following files from `Desktop/EMG_data/YYYYMMDD/x_train_data/`:

```text
x_train_all.npy
peak_data_grip_data.npy
peak_data_wrist_extension_data.npy
```

`x_train_all.npy` は、以下の順番で正確に3行の信号を格納している必要があります。

`x_train_all.npy` must contain exactly three rows in the following order:

| Class label | 日本語 | English |
| ---: | --- | --- |
| 0 | 握り | Grip |
| 1 | 反り | Wrist extension |
| 2 | 動作なし | No movement |

出力形状は次のようになります。

The output shape is:

```text
(3 × samples_per_class, 2 × half_window)
```

`sample_count`（1クラス当たりのサンプル数）と `half_window` を、例えば `5 10` のようにスペース区切りで入力し、`y` で確定します。この例では15行・20特徴量になります。動作なしは位置200から1点ずつ開始位置をずらして切り出すため、隣接サンプルは重複します。信号の大きさに関する警告が出た場合は、保存時に追加で `y` の確認が必要です。

Enter `sample_count` (samples per class) and `half_window` separated by a space, for example `5 10`, then confirm with `y`. This produces 15 rows and 20 features. No-movement windows start at index 200 and advance one value at a time, so adjacent samples overlap. If a signal-level warning appears, saving requires an additional `y` confirmation.

### `training_emg_mlp_pytorch.py`

前処理済みのEMG訓練配列を使用して、全結合多層パーセプトロン（MLP）を学習するスクリプトです。

This script trains a fully connected multilayer perceptron (MLP) using the prepared EMG training array.

入力には、行をサンプル、列を特徴量とする2次元の `.npy` 配列を使用します。総行数は動作クラス数で割り切れる必要があります。

The input must be a two-dimensional `.npy` array whose rows are samples and whose columns are features. The total number of rows must be divisible by the number of movement classes.

各クラスのサンプルが同数ずつ連続して並んでいることを前提とし、行の並び順から `0, 1, ...` のクラスラベルを自動生成します。`prepare_emg_training_data.py` の出力を使用する場合は、実行後の動作数の入力欄に `3` を入力してください。

The script assumes that every class has the same number of consecutive rows. It generates class labels automatically as `0, 1, ...` from the row order. When using output from `prepare_emg_training_data.py`, enter `3` at the movement-count prompt after starting the script.

学習ファイル名は `x_train_<総サンプル数>_2x<half_window>.npy` の形式にしてください。ファイル名から総サンプル数と片側のデータ数を読み取り、配列形状が `(総サンプル数, 2 * half_window)` と一致するか確認します。例えば `x_train_15_2x10.npy` は `(15, 20)` の配列が必要です。ファイル名を任意に変更したり、読み込み前にクラスごとの行順を崩したりしないでください。

Training files must follow the naming pattern `x_train_<total_samples>_2x<half_window>.npy`. The script parses the total sample count and half-window size from the filename and checks that the array shape matches `(total_samples, 2 * half_window)`. For example, `x_train_15_2x10.npy` must have shape `(15, 20)`. Do not rename the file arbitrarily or shuffle class groups before loading it.

ネットワークは、ReLUとDropoutを持つ2つの全結合隠れ層と、出力層で構成されます。損失関数には `CrossEntropyLoss`、最適化手法にはSGDを使用します。

The network consists of two hidden fully connected layers with ReLU and Dropout, followed by an output layer. Training uses `CrossEntropyLoss` and the SGD optimizer.

既定の学習条件:

Default training settings:

| 設定 / Setting | 既定値 / Default |
| --- | ---: |
| 隠れ層のユニット数 / Hidden dimension | `600` |
| Dropout | `0.5` |
| 学習率 / Learning rate | `0.01` |
| エポック数 / Epochs | `2000` |
| バッチサイズ / Batch size | `8` |
| 途中保存間隔 / Save interval | `100` |
| 表示間隔 / Log interval | 各訓練バッチ / Every training batch |
| 乱数シード / Random seed | 固定なし / Not fixed |
| デバイス / Device | `cpu` |

学習はCPUで実行します。設定はコード内の `hidden_dim`、`num_epochs`、`batch_size`、最適化処理で定義されています。

Training runs on the CPU. Settings are defined in the code (`hidden_dim`, `num_epochs`, `batch_size`, and the optimizer). 

学習済みモデルは、PyTorchの `state_dict` として `.pth` 形式で、100エポックごとと学習終了時に同じファイルへ保存されます。学習条件のJSONファイルは生成しません。学習時間は `Desktop/log/PyTorch_learning_time.txt` に追記されます。

The trained model is saved as a PyTorch `state_dict` with the `.pth` extension, every 100 epochs and at the end of training, to the same file. No JSON metadata is generated. Training time is appended to `Desktop/log/PyTorch_learning_time.txt`.

### `emg_motion_classification_cursor_control.py`

当日の `DL_model` フォルダから学習済み `.pth` モデルを読み込み、受信したARVをCPUで識別します。右腕は `r`、左腕は `l` を入力し、モデルを選択します。入力次元と隠れ層の大きさは保存された重みから取得します。入力特徴量数は偶数、出力は3クラスが必要です。

This script loads a trained `.pth` model from today's `DL_model` folder and classifies incoming ARV values on the CPU. Enter `r` for the right arm or `l` for the left arm, then select a model. Input and hidden dimensions are read from the saved weights; the input feature count must be even and the model must have three output classes.

入力窓がたまった後、5入力ごとに予測します。動作終了時に予測系列をルールで処理し、矢印キー入力または水平・垂直モードの切替を行います。送信するのはマウスポインタの座標ではなく、キーボードの矢印キーです。操作対象アプリにフォーカスを合わせてください。終了時は実行中のコンソールで `Ctrl+C` を押します。

After the input window fills, predictions are made every five input values. At movement completion, rules applied to the prediction sequence determine an arrow-key action or a switch between horizontal and vertical modes. The program sends keyboard arrow keys, not mouse-pointer coordinates. Focus the target application to receive the keys. Press `Ctrl+C` in the running console to stop.

| 計測する腕 / Arm | モード / Mode | 握り / Grip | 反り / Wrist extension |
| --- | --- | --- | --- |
| 左腕 / Left | 水平 / Horizontal | → | ← |
| 左腕 / Left | 垂直 / Vertical | ↑ | ↓ |
| 右腕 / Right | 水平 / Horizontal | ← | → |
| 右腕 / Right | 垂直 / Vertical | ↓ | ↑ |

初期状態は水平モードで、モード切替は移動より優先されます。3ラベルすべてを含む系列の長さが、両腕とも16（既定の `mean_movement_length - 3`）を超えると切り替わります。キャリブレーション用フォルダは作成されますが、自動キャリブレーションは実装されていません。

The initial mode is horizontal. Mode switching takes priority over movement. For both arms, a sequence containing all three labels switches modes when its length exceeds 16 (the default mean_movement_length - 3). A calibration folder is created, but automatic calibration is not implemented.

## 必要な環境 / Requirements

このプログラムはWindows環境での使用を想定しています。

This program is intended to run on Windows.

Python 3.11 の使用を推奨します。

Python 3.11 is recommended.

5つのスクリプトでは次の外部パッケージを使用します。Python環境にインストールし、SDK固有の依存関係は別途追加してください。

The five scripts use the following third-party packages. Install them in your Python environment; add SDK-specific dependencies separately.

```bat
python -m pip install numpy scipy matplotlib pygame torch keyboard
```

`prepare_emg_training_data.py` ではNumPyを使用します。  
`training_emg_mlp_pytorch.py` ではNumPyとPyTorchを使用します。今回の学習・推論コードはCPUで実行し、GPUを選択する処理はありません。

`prepare_emg_training_data.py` requires NumPy.  
`training_emg_mlp_pytorch.py` requires NumPy and PyTorch. The supplied training and inference code runs on the CPU and does not implement GPU selection.

## 音声ファイル / Audio Files

このプログラムでは、Windowsのデスクトップ上にある `audio` フォルダ内の音声ファイルを使用する想定です。

This program assumes that audio cue files are placed in the `audio` folder on the Windows Desktop.

想定しているファイルパスは以下の通りです。

Expected file paths:

```text
Desktop/audio/nigiri.wav
Desktop/audio/sori.wav
Desktop/audio/finish.wav
Desktop/audio/start.wav
```

これらの音声ファイルは動作タイミングの通知に使い、`start.wav` はリアルタイム識別の開始通知に使用します。音声ファイルは今回の提供内容には含まれていません。カーソル操作プログラムはWindows標準モジュールの `winsound` も使用します。

These audio files provide movement cues; `start.wav` signals the start of real-time classification. Audio files are not included in the supplied set. The cursor-control script also imports the Windows-only standard module `winsound`.

## パラメータ / Parameters

主な実験条件は `collect_pretraining_emg_data.py` 内の定数として定義しています。

Main experimental parameters are defined as constants in `collect_pretraining_emg_data.py`.

```python
INITIAL_GRIP_EXCLUSION_SAMPLES = 451
INITIAL_WRIST_EXTENSION_EXCLUSION_SAMPLES = 451
INITIAL_NO_MOVEMENT_EXCLUSION_SAMPLES = 601
NO_MOVEMENT_REPEAT_COUNT = 8
SAMPLES_PER_MOVEMENT = 4100
```

ピーク検出のしきい値は、私たちの実験環境に合わせて調整した値です。

Peak detection thresholds were tuned for our experimental environment.

使用するセンサ、被験者、実験環境に応じて、これらの値を調整してください。

Please adjust these parameters depending on your sensor, subject, and experimental environment.

## 出力 / Output

取得したデータは、以下のディレクトリに保存されます。

The collected data is saved under the following directory:

```text
Desktop/EMG_data/YYYYMMDD/x_train_data/
```

出力されるファイルの例です。

Example output files:

```text
x_train_all.npy
peak_data_grip_data.npy
peak_data_wrist_extension_data.npy
```

前処理済み訓練データは、既定では同じデータフォルダに保存されます。

Prepared training data is saved in the same data directory by default:

```text
Desktop/EMG_data/YYYYMMDD/x_train_data/
└── x_train_<total_samples>_2x<half_window>.npy
```

たとえば `sample_count = 5`、`half_window = 10` の場合は `x_train_15_2x10.npy` が作成されます。同じ設定で再実行すると上書きされます。

For example, `sample_count = 5` and `half_window = 10` produce `x_train_15_2x10.npy`. Running the script again with the same settings overwrites this file.

モデルは `Desktop/EMG_data/YYYYMMDD/DL_model/` に、`train_FC_<class_count>_<sample_count>_<half_window>_<YYYYMMDD_HHMMSS>.pth` の形式で保存されます。3クラス・15サンプル・20特徴量の場合、名前は `train_FC_3_15_10_` で始まります。3つ目の数値は入力特徴量数ではなく片側のデータ数です。

Models are saved under `Desktop/EMG_data/YYYYMMDD/DL_model/` as `train_FC_<class_count>_<sample_count>_<half_window>_<YYYYMMDD_HHMMSS>.pth`. For 3 classes, 15 samples, and 20 features, the filename begins with `train_FC_3_15_10_`: the third number is the half-window size, not the input feature count. 

`.npy`, `.csv`, `.png` などの生成データやログファイルは、通常GitHubにはアップロードしないでください。

Generated data files such as `.npy`, `.csv`, `.png`, and log files should not usually be uploaded to GitHub.

## 著者 / Authors

本リポジトリは、伊藤正剛および北本卓也により作成されました。

This repository was prepared by Masataka Ito and Takuya Kitamoto.

## ライセンス / License

このプロジェクトはMITライセンスのもとで公開されています。詳細は [LICENSE](LICENSE) ファイルを確認してください。

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 免責事項 / Disclaimer

本ソフトウェアは、研究および教育目的で提供されています。

This software is provided for research and educational purposes.

本ソフトウェアの使用によって生じたいかなる問題、損害、データ損失についても、作者は責任を負いません。

The authors are not responsible for any issues, damages, or data loss caused by the use of this software.
