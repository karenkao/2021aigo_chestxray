# 2021 AIGO chestxray competition
2021 AIGO competition of pneumothorax and chest-tube classification based on chest X-Ray

## 目的
開發一個足夠精確的classification model，並以CAM （Class Activation Map） 呈現氣胸與胸管位置。

## 方法
- 氣胸分類模型：使用Densenet169模型將影像輸入模型並做二元分類，預測有/無氣胸。
- 胸管分類模型：使用Densenet169模型將影像輸入模型並做二元分類，預測有/無胸
- 可視化模型：使用Grad-CAM，將氣胸及胸管位置可視化於xray影像上。

## 模型成效

## 部署
