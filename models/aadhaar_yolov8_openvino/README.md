# Inference
## Supported Labels
```python
# label_id: label_name
{0: "AADHAR_NUMBER", 1: "DATE_OF_BIRTH", 2: "GENDER", 3: "NAME", 4: "ADDRESS"}
```

## Install Dependencies
```
$ pip install ultralytics huggingface_hub supervision
```

## Load model
```python
from ultralytics import YOLO
from huggingface_hub import snapshot_download
from supervision import Detections

MODEL_REPO_ID = "thewalnutaisg/YOLOv8-aadhar-card-int8-openvino"
MODEL_LOCAL_DIR = "./models/model_int8_openvino_model"

snapshot_download(
    repo_id=MODEL_REPO_ID,
    local_dir=MODEL_LOCAL_DIR,
)

image_url = "./Sample 1/100001005885477 (1).jpg"

model = YOLO(MODEL_LOCAL_DIR, task="detect")
detections = Detections.from_ultralytics(model.predict(image_url)[0])

print(detections)
```