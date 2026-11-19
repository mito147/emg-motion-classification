#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本コードは、実験で使用したコードをもとに公開用として整理したものです。
This public version is based on the code used in the experiments.
可読性向上のため、一部の変数名を変更しています。
Some variable names have been changed to improve readability.
ファイルおよびフォルダのパス指定も整理しています。
File and directory path definitions have also been simplified.
未使用部分や、一部の時間計測・ログ出力に関する処理を削除しています。
Unused code and some timing and logging routines have been removed.
センサSDK固有の通信コードは、再配布に関する制約のため本公開版には含めていません。
Sensor SDK-specific communication code is omitted from this public version due to redistribution restrictions.
実際にデータを取得する場合は、使用するセンサおよびSDKに合わせて通信処理を実装してください。
To collect data, implement the communication routines appropriate for your sensor and SDK.
実験の再現性を保つため、主要なデータ処理の流れおよび実験・学習ロジックは維持しています。
The core data-processing flow and experimental and training logic have been retained to preserve experimental reproducibility.
本コードは参考実装であり、そのまま実行できる完成済みアプリケーションではありません。
This code is a reference implementation rather than a standalone, ready-to-run application.
"""

import datetime
import sys
import threading
import time
import traceback
import winsound
from collections import deque
from pathlib import Path

import keyboard
import numpy as np
import pygame
import torch
import torch.nn as nn


# ファイルとフォルダのパス設定（環境に合わせて変更してください）。
# File and directory paths (adjust these for your environment).
DESKTOP_DIR = Path.home() / "Desktop"
DATA_ROOT = DESKTOP_DIR / "EMG_data"
AUDIO_DIR = DESKTOP_DIR / "audio"

# 当日のデータフォルダ内にある学習済みモデルを使用します。
# Use trained models from today's data directory.
DATA_DATE = datetime.date.today().strftime("%Y%m%d")
DATA_DIR = DATA_ROOT / DATA_DATE
MODEL_DIR = DATA_DIR / "DL_model"
CALIBRATION_DIR = DATA_ROOT / "calibration"
START_SOUND_PATH = AUDIO_DIR / "start.wav"


# モデル定義 (EMGModelクラスが既に定義されていると仮定)
# Model definition (assuming the EMGModel class is already defined).
class EMGModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_count):
        """
        筋電位の動作識別に使用する全結合MLPを初期化します。
        Initialize a fully connected MLP for EMG motion classification.

        2つの隠れ層にReLUとDropoutを使用し、最後にクラス別の出力を生成します。
        Use ReLU and Dropout in two hidden layers, followed by a class output layer.

        Args:
            input_dim (int): 1サンプルに含まれる入力特徴量数。
                Number of input features per sample.
            hidden_dim (int): 各隠れ層のユニット数。
                Number of units in each hidden layer.
            class_count (int): 識別する動作クラス数。
                Number of movement classes to classify.
        """
        super(EMGModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_dim, class_count)

    def forward(self, x):
        """
        入力データから各動作クラスのロジットを計算します。
        Compute logits for each movement class from the input data.

        出力は確率ではなく、argmaxによるクラス選択に使用する値です。
        The outputs are logits used for class selection with argmax, rather than probabilities.

        Args:
            x (torch.Tensor): 形状が (バッチサイズ, input_dim) の入力データ。
                Input data with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: 形状が (バッチサイズ, class_count) のロジット。
                Logits with shape (batch_size, class_count).
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

HORIZONTAL = 0
VERTICAL = 1

def handle_cursor_action(arm_side,actionSignal,movement_predictions,cursor_mode,mean_movement_length):
  """
  1動作中の予測結果に基づいて、カーソル移動または操作モードの切替を行います。
  Move the cursor or switch the control mode based on predictions collected during a movement.

  計測する腕と操作モードに応じて、矢印キーを送信します。
  Send arrow-key events according to the measured arm and the current control mode.

  モード切替条件を満たした場合は、通知音を鳴らして水平・垂直モードを切り替えます。
  Play a notification beep and switch between horizontal and vertical modes when the switching condition is met.

  Args:
      arm_side (str): 計測する腕。左腕は "l"、右腕は "r"。
          Measured arm: "l" for the left arm or "r" for the right arm.
      actionSignal (set[int]): 1動作中の予測系列に含まれるクラス番号の集合。
          Set of class labels appearing in the prediction sequence for one movement.
      movement_predictions (deque): 1動作中に蓄積された予測クラスの系列。
          Sequence of predicted class labels accumulated during one movement.
      cursor_mode (int): 現在の操作モード。HORIZONTAL または VERTICAL。
          Current control mode: HORIZONTAL or VERTICAL.
      mean_movement_length (float): 左腕のモード切替判定に使用する長さの基準値。
          Reference sequence length used for the left-arm mode-switching condition.

  Returns:
      int | None: 更新後の操作モード。腕やモードが対応外の場合は None。
          Updated control mode, or None if the arm or mode is unsupported.
  """
  if arm_side == 'l':
    if 0 in actionSignal and 1 in actionSignal and 2 in actionSignal and len(movement_predictions) > mean_movement_length - 3 :
      note_freq = 440
      note_duration = 500
      print("動作モード変更\nSwitching cursor mode.")
      print("動作モード変更\nSwitching cursor mode.")
      print("動作モード変更\nSwitching cursor mode.")
      print("動作モード変更\nSwitching cursor mode.")
      print("動作モード変更\nSwitching cursor mode.")
      print("動作モード変更\nSwitching cursor mode.")
      print("動作モード変更\nSwitching cursor mode.")
      winsound.Beep(note_freq, note_duration)
      if cursor_mode == HORIZONTAL:
          cursor_mode = VERTICAL
      elif cursor_mode == VERTICAL:
          cursor_mode = HORIZONTAL
      return cursor_mode

    if cursor_mode == HORIZONTAL:
        
      if 0 in actionSignal and 1 in actionSignal and 2 in actionSignal:
        print("握っています\nGripping.")
        keyboard.press_and_release("right")
                    
      elif 1 in actionSignal and 2 in actionSignal:
        print("反っています\nExtending the wrist.")
        keyboard.press_and_release("left")
        print(actionSignal)
        
      return HORIZONTAL
  
    if cursor_mode == VERTICAL:
      if 0 in actionSignal and 1 in actionSignal and 2 in actionSignal:
        print("握っています\nGripping.")
        print(actionSignal)
        keyboard.press_and_release("up")
    
      elif  1 in actionSignal and 2 in actionSignal:
        print("反っています\nExtending the wrist.")
        keyboard.press_and_release("down")
        print(actionSignal)
      
      return VERTICAL

  if arm_side == 'r':
    #右腕
    # Right arm.
    if 0 in actionSignal and 1 in actionSignal and 2 in actionSignal and len(movement_predictions) > mean_movement_length - 3:
      note_freq = 440
      note_duration = 500
      print("動作モード変更\nSwitching cursor mode.")
      print("動作モード変更\nSwitching cursor mode.")
      print("動作モード変更\nSwitching cursor mode.")
      print("動作モード変更\nSwitching cursor mode.")
      print("動作モード変更\nSwitching cursor mode.")
      print("動作モード変更\nSwitching cursor mode.")
      print("動作モード変更\nSwitching cursor mode.")
      winsound.Beep(note_freq, note_duration)
      if cursor_mode == HORIZONTAL:
          cursor_mode = VERTICAL
      elif cursor_mode == VERTICAL:
          cursor_mode = HORIZONTAL
      return cursor_mode

    if cursor_mode == HORIZONTAL:
        
      if 0 in actionSignal and 1 in actionSignal and 2 in actionSignal:
        print("握っています\nGripping.")
        print(actionSignal)
        keyboard.press_and_release("left")
        
      elif  1 in actionSignal and 2 in actionSignal:
        print("反っています\nExtending the wrist.")
        keyboard.press_and_release("right")
        print(actionSignal)
        
      return HORIZONTAL
  
    if cursor_mode == VERTICAL:
      if 0 in actionSignal and 1 in actionSignal and 2 in actionSignal:
        print("握っています\nGripping.")
        print(actionSignal)
        keyboard.press_and_release("down")
    
      elif  1 in actionSignal and 2 in actionSignal:
        print("反っています\nExtending the wrist.")
        keyboard.press_and_release("up")
        print(actionSignal)
      
      return VERTICAL

def list_files_and_select(folder_path):
  """
  指定フォルダ内の学習済みモデルを一覧表示し、利用者が選択したパスを返します。
  List trained models in the specified directory and return the path selected by the user.

  .pthファイルだけを対象とし、更新日時の新しい順に表示します。
  Include only .pth files and display them in descending modification-time order.

  Args:
      folder_path (str | Path): 学習済みモデルを格納したフォルダ。
          Directory containing trained model files.

  Returns:
      Path: 利用者が選択した学習済みモデルのパス。
          Path to the trained model selected by the user.
  """
  files = [
      path
      for path in Path(folder_path).iterdir()
      if path.is_file() and path.suffix.lower() == ".pth"
  ]  
  
  if not files:
      raise FileNotFoundError(
          f"学習済みモデルが見つかりません: {folder_path}\n"
          f"No trained model files were found: {folder_path}"
      )
  
  # ファイルを最終更新時刻で降順にソート
  # Sort files by modification time, newest first.
  files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
  
  # ファイル名と最終更新時刻を表示
  # Display file names and modification times.
  for i, file in enumerate(files):
    timestamp = file.stat().st_mtime
    formatted_time = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    print(f"{i + 1}: {formatted_time} - {file.name}")
    
  # ユーザーにファイルの選択を促す
  # Prompt the user to select a file.
  while True:
    try:
      selected_index = int(input("選択したいファイルの番号を入力してください:\nEnter the number of the file to select: ")) - 1
      if 0 <= selected_index < len(files):
        selected_file = files[selected_index]
        print(f"選択されたファイル: {selected_file}\nSelected file: {selected_file}")
        return selected_file
      else:
        print(f"エラー: 番号は1から{len(files)}の範囲で入力してください。\nError: Enter a number from 1 to {len(files)}.")
    except ValueError:
      print("エラー: 数字を入力してください。\nError: Please enter a number.")

def initialize_pygame():
    """
    音声再生に使用するpygameのmixerを初期化します。
    Initialize the pygame mixer used for audio playback.

    音声ファイルを再生する前に呼び出してください。
    Call this function before playing audio files.

    Returns:
        None
    """
    pygame.mixer.init()


def play_sound_in_thread(sound_path):
    """
    音声ファイルの再生処理を別スレッドで開始します。
    Start audio-file playback in a separate thread.

    デーモンスレッドを使用し、再生終了を待たずに呼び出し元へ戻ります。
    Use a daemon thread and return without waiting for playback to finish.

    Args:
        sound_path (str | Path): 再生する音声ファイルのパス。
            Path to the audio file to play.

    Returns:
        None
    """
    def play_sound():
        """
        外側の関数で指定された音声ファイルを読み込み、再生します。
        Load and play the audio file specified by the enclosing function.

        このスレッド内で再生終了を待ちます。
        Wait for playback to finish within this thread.

        Returns:
            None
        """
        pygame.mixer.music.load(str(sound_path))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    threading.Thread(target=play_sound, daemon=True).start()


def serial_communication_predict(Ser, arm_side, model, half_window, folder_path_calibration):
    """
    ARVデータを逐次入力し、動作識別とカーソル操作を繰り返します。
    Process incoming ARV values repeatedly to classify movements and control the cursor.

    2 * half_window個の値を入力窓とし、5個の入力ごとに予測します。
    Use an input window of 2 * half_window values and predict every five input values.

    クラス2を動作なしとして、動作の開始と終了を判定します。
    Treat class 2 as no movement to detect movement onset and completion.

    動作終了時に予測系列をカーソル操作関数へ渡します。
    Pass the accumulated prediction sequence to the cursor-control function when a movement ends.

    SDK固有の受信処理は省略しているため、ARVを取得する処理を追加してください。
    SDK-specific reception is omitted; add the code that obtains each ARV value.

    Args:
        Ser: センサSDKの通信オブジェクト。公開版では通信処理を省略しています。
            Sensor SDK communication object; communication routines are omitted from the public version.
        arm_side (str): 計測する腕。左腕は "l"、右腕は "r"。
            Measured arm: "l" for the left arm or "r" for the right arm.
        model (EMGModel): 評価モードに設定済みの学習済みモデル。
            Trained model already set to evaluation mode.
        half_window (int): 入力窓に含まれる値の数の半分。
            Half the number of values in the input window.
        folder_path_calibration (str | Path): キャリブレーション用のパス引数。現在は未使用。
            Calibration-path argument; currently unused.

    Returns:
        通常は戻らず、割り込みまたはエラーが発生するまで処理を続けます。
        Normally does not return; processing continues until an interrupt or error occurs.
    """
    input_buffer = deque()
    recent_predictions = deque(maxlen=2)
    movement_predictions = deque()
    sample_count = 0
    movement_active = False
    cursor_mode = HORIZONTAL
    calibration_count = 0
    mean_movement_length = 19
    
    play_sound_in_thread(START_SOUND_PATH)
    
    while True:
        try:
            time.sleep(0.01)
            
            input_buffer.append(ARV)
            
            if sample_count >= 2 * half_window:
                input_buffer.popleft()
                if sample_count % 5 == 0:
                    input_array = np.array(input_buffer, dtype=np.float32)
                    input_array = input_array.reshape(1, 2 * half_window)
                    input_tensor = torch.tensor(input_array)
                        
                    with torch.no_grad():
                        model_output = model(input_tensor)
                        predicted_class = torch.argmax(model_output, dim=1).item()

                    recent_predictions.append(predicted_class)
                        
                    if len(recent_predictions) >= 2:
                        if recent_predictions[0] == 2 and recent_predictions[1] != 2:
                            movement_active = 1
                            movement_predictions.append(predicted_class)

                        if movement_active:
                            movement_predictions.append(predicted_class)
                            print("判別した値->", predicted_class)
                            print("Predicted class ->", predicted_class)
                            print(movement_predictions)

                        if recent_predictions[0] != 2 and recent_predictions[1] == 2:
                            
                            movement_active = 0
                            actionSignal = set(movement_predictions)
                            print("set->", actionSignal)
                            
                            cursor_mode = handle_cursor_action(arm_side, actionSignal, movement_predictions, cursor_mode, mean_movement_length)
                        
                            movement_predictions.clear()
                        
                        if len(movement_predictions) > 24:
                            movement_predictions.clear()
            sample_count += 1

        except KeyboardInterrupt:
            # 割り込みを外側の終了処理へ伝えます。
            # Propagate the interrupt to the outer shutdown handler.
            raise

        except Exception:
            # エラーを外側へ伝え、詳細を表示して終了します。
            # Propagate the error so the outer handler can report it and exit.
            raise


initialize_pygame()

while True:
    print(
        "右腕での計測ならrを、左腕での計測ならlを入力してください。\n"
        "Enter 'r' for measurements on the right arm "
        "or 'l' for measurements on the left arm."
        )
    arm_side=input()
    
    if arm_side == 'r':
        print("OK --> Right")
        break
    elif arm_side == 'l':
        print("OK --> Left")
        break
    else:
        continue
try:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    # 学習済みモデルのファイルを選択します。
    # Select a trained model file.
    selected_file = list_files_and_select(MODEL_DIR)
            
        # 学習済みモデルの重みをCPUに読み込みます。
    # Load the trained model weights onto the CPU.
    state_dict = torch.load(
        selected_file,
        map_location="cpu",
        weights_only=True,
    )

    # 保存された重みからモデルの構成を取得します。
    # Determine the model dimensions from the saved weights.
    hidden_dim, input_dim = state_dict["fc1.weight"].shape
    class_count = state_dict["fc3.weight"].shape[0]

    if input_dim % 2 != 0:
        raise ValueError(
            "入力特徴量数は偶数である必要があります。\n"
            "The number of input features must be even."
        )

    if class_count != 3:
        raise ValueError(
            "3クラスの学習済みモデルを使用してください。\n"
            "Use a trained model with three classes."
        )

    # 既存の入力バッファ処理に合わせ、入力特徴量数の半分を設定します。
    # Use half the input feature count for the existing buffer logic.
    half_window = input_dim // 2
    class_count = state_dict["fc3.weight"].shape[0]

    model = EMGModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        class_count=class_count,
    )
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Deep Learningモデルの処理を実行します。フォルダパス: {selected_file}\nRunning the deep learning model. Path: {selected_file}")

    # シリアル通信の予測開始
    # Start prediction using serial communication.
    serial_communication_predict(Ser, arm_side, model, half_window, DATA_DIR)
    
        
    
except KeyboardInterrupt:
    print("キーボード割り込みが発生しました。\nKeyboard interrupt detected.")
    time.sleep(2)
    
except Exception as e:
    print(f"エラーが発生しました: {e}\nAn error occurred: {e}")
    traceback.print_exc()
    
finally:
    sys.exit(0)

