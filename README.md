# 2021 AIGO chestxray competition
2021 AIGO competition of pneumothorax and chest-tube classification based on chest X-Ray

## 目的
開發一個足夠精確的classification model，並以CAM （Class Activation Map） 呈現氣胸與胸管位置。

## 方法
- 氣胸分類模型：使用Densenet201模型與ImageNet預訓練權重將影像輸入模型並做二元分類，預測有/無氣胸。訓練時使用十倍的資料增強(Data augmentation)，並使用dropout與L2 regularization以避免模型過擬合(overfitting)。
- 胸管分類模型：使用Densenet201模型將影像輸入模型並做二元分類，預測有/無胸管，訓練時使用CLAHE提高影像對比與十倍的資料增強(Data augmentation)。此外，使用dropout與L2 regularization以避免模型過擬合(overfitting)。
- 可視化模型：使用Grad-CAM，將氣胸及胸管位置可視化於xray影像上。

## 模型成效
- 氣胸分類模型
- 胸管分類模型(測試資料集:288張胸腔x光影像)
  - 混淆矩陣(confusion matrix
  <img width="388" alt="tube_confusion" src="https://user-images.githubusercontent.com/44295049/137432791-d4fb767e-42cb-4ee9-a46f-344cf6b1f405.png">
  - AUC達 0.99

## 部署
使用Flask 開發Web API。
- 輸入
- 輸出
