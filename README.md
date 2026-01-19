# Real-Time AI Exercise Form Correction Using YOLO and MediaPipe

### Description
A Study on the big three exercises AI posture correction service Using YOLOv5 and MediaPipe<br>

## Data
- YOLOv5: Detect only one Person
  - Scraping Images from Google & Roboflow
    - Bench Press
      - Google Search Keyword: Bench Press
      - Faller Computer Vision Project with lying down
    - Squat
      - Squat-Depth Image Dataset
      - HumonBody1 Computer Vision Project with Standing
    - Deadlift
      - SDT Image Dataset
    - More(bending, lying, sitting, standing)
      - Silhouettes of human posture
  
## Training & Evaluation
### YOLOv5
  - Detect only a Person exercising something
    - Hyperparameters to train
      - epochs 200(but early stopping: 167)
      - batch 16
      - weights yolov5s.pt
      - etc are set by 'default'
  - Performance Evaluation
    |Precision|Recall|mAP_0.5|mAP_0.5:0.95|
    |:--:|:--:|:--:|:--:|
    |0.987|0.990|0.99|0.686|
    
### Exercise Classfication
  - Bench Press (Algorithm: Random Forest)
    |Accuracy|Precision|Recall|F1-Score|
    |:--:|:--:|:--:|:--:|
    |0.961|0.963|0.961|0.961|
  - Squat (Algorithm: Random Forest)
    |Accuracy|Precision|Recall|F1-Score|
    |:--:|:--:|:--:|:--:|
    |0.989|0.989|0.989|0.989|
  - Deadlift (Algorithm: Random Forest)
    |Accuracy|Precision|Recall|F1-Score|
    |:--:|:--:|:--:|:--:|
    |0.947|0.949|0.947|0.948|


## How to Use
- Open your terminal in mac, linux or your command prompt in Windows. Then, type "<b>Streamlit run Streamlit.py</b>".<br>

