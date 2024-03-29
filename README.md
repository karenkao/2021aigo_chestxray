# 2021 AIGO chestxray competition
2021 AIGO competition of pneumothorax and chest-tube classification based on chest X-Ray

## 目的
開發一個足夠精確的classification model，並以CAM （Class Activation Map） 呈現氣胸與胸管位置。

## 方法
- 氣胸分類模型：使用Densenet201模型與ImageNet預訓練權重將影像輸入模型並做二元分類，預測有/無氣胸。訓練時使用十倍的資料增強(Data augmentation)，並使用dropout與L2 regularization以避免模型過擬合(overfitting)。
- 胸管分類模型：使用Densenet201模型將影像輸入模型並做二元分類，預測有/無胸管，訓練時使用CLAHE提高影像對比與十倍的資料增強(Data augmentation)。此外，使用dropout與L2 regularization以避免模型過擬合(overfitting)。
- 可視化模型：使用Grad-CAM，將氣胸及胸管位置可視化於xray影像上。

## 模型成效
- 氣胸分類模型(測試資料集:1175張胸腔x光影像)
  - 混淆矩陣(confusion matrix)<br>
  <img width="365" alt="pneumo_cm" src="https://user-images.githubusercontent.com/44295049/137433836-16b13848-d1fa-4a44-9b24-da1bd74b604f.png">

  - AUC達 0.98
  <img width="349" alt="pneumo_auc" src="https://user-images.githubusercontent.com/44295049/137433852-57f7eee2-061c-4209-a102-075e06daa14d.png">

- 胸管分類模型(測試資料集:288張胸腔x光影像)
  - 混淆矩陣(confusion matrix)<br>
  <img width="388" alt="tube_confusion" src="https://user-images.githubusercontent.com/44295049/137432791-d4fb767e-42cb-4ee9-a46f-344cf6b1f405.png">
  
  - AUC達 0.99
  
  <img width="338" alt="tube_auc" src="https://user-images.githubusercontent.com/44295049/137433302-59e3c93a-3aec-43ee-93ff-503778558e35.png">

## 部署
使用Flask 開發Web API。參考call_api_demo.ipynb.
- 輸入: RBG影像encode為base64 string，並以POST {"img_str": img_str} 傳入url: http://127.0.0.1:8080/predict
- 輸出: response json檔案，格式為 {"tube": {"pred": 0.99, "mask": img_base64_str}, "pneumo":{"pred": 0.99, "mask": img_base64_str}}

<img width="218" alt="pneumo_pred" src="https://user-images.githubusercontent.com/44295049/137435115-cac41404-863a-4b96-b4ed-2ed1d1190106.png">
<img width="211" alt="tube_pred" src="https://user-images.githubusercontent.com/44295049/137435116-11167051-11a0-4531-93df-cdd6d7d75462.png">

