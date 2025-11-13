# YOLOv5 + Re-ID Integration - Quick Start Guide

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/SaiVijay24/Yolo_v5_sample_integration
cd Yolo_v5_sample_integration

# Install dependencies
pip install -r requirements.txt
```

## 📦 What You Need

### For Basic Object Detection:
- YOLOv5 model weights (e.g., `yolov5s.pt`) - downloads automatically
- Input images/videos

### For Re-Identification (Current Setup):
- ✅ `model_jit.pth` - Pre-trained re-ID model (must be in repository root)
- ⚠️ Note: Currently uses dummy embeddings - replace for production

---

## 🎯 Common Tasks

### 1. Detect Objects in an Image
```bash
python detect.py --source path/to/image.jpg --weights yolov5s.pt
```
**Output**: Results saved to `runs/detect/exp/`

### 2. Detect Objects in a Video
```bash
python detect.py --source path/to/video.mp4 --weights yolov5s.pt
```

### 3. Use Webcam for Real-time Detection
```bash
python detect.py --source 0 --weights yolov5s.pt
```
**Note**: `0` refers to the default webcam

### 4. Detect Only Specific Objects (e.g., Persons)
```bash
python detect.py --source image.jpg --weights yolov5s.pt --classes 0
```
**Class IDs**: 0=person, 16=dog, 2=car, etc. (COCO classes)

### 5. Save Detection Results to Text Files
```bash
python detect.py --source image.jpg --weights yolov5s.pt --save-txt
```
**Output**: Bounding box coordinates in `runs/detect/exp/labels/`

### 6. Visualize Results in Real-time
```bash
python detect.py --source image.jpg --weights yolov5s.pt --view-img
```

### 7. Save Cropped Detections
```bash
python detect.py --source image.jpg --weights yolov5s.pt --save-crop
```
**Output**: Cropped images in `runs/detect/exp/crops/`

---

## 🎓 Training Your Own Model

### Step 1: Prepare Dataset
Organize your dataset in YOLO format:
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Step 2: Create Dataset Configuration
Create `data/custom_data.yaml`:
```yaml
train: dataset/images/train
val: dataset/images/val
nc: 2  # number of classes
names: ['class1', 'class2']
```

### Step 3: Start Training
```bash
python train.py --data data/custom_data.yaml --weights yolov5s.pt --epochs 100 --batch-size 16
```

### Step 4: Monitor Training
Results and logs saved to `runs/train/exp/`

---

## 🔍 Model Selection Guide

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| **yolov5n** | ⚡⚡⚡⚡⚡ | ⭐⭐ | Mobile devices, edge computing |
| **yolov5s** | ⚡⚡⚡⚡ | ⭐⭐⭐ | **Recommended for most users** |
| **yolov5m** | ⚡⚡⚡ | ⭐⭐⭐⭐ | Good balance of speed and accuracy |
| **yolov5l** | ⚡⚡ | ⭐⭐⭐⭐⭐ | High accuracy requirements |
| **yolov5x** | ⚡ | ⭐⭐⭐⭐⭐ | Maximum accuracy, research |

---

## 🎛️ Important Parameters

### Detection Parameters

| Parameter | Default | Description | Example |
|-----------|---------|-------------|---------|
| `--source` | `data/images` | Input source (image, video, webcam, URL) | `--source video.mp4` |
| `--weights` | `yolov5s.pt` | Model weights file | `--weights yolov5m.pt` |
| `--conf-thres` | `0.25` | Confidence threshold (0-1) | `--conf-thres 0.5` |
| `--iou-thres` | `0.45` | NMS IoU threshold | `--iou-thres 0.5` |
| `--classes` | None | Filter by class(es) | `--classes 0 1 2` |
| `--img-size` | `640` | Inference image size | `--img-size 1280` |
| `--save-txt` | False | Save results to *.txt | `--save-txt` |
| `--save-crop` | False | Save cropped detections | `--save-crop` |
| `--nosave` | False | Don't save images/videos | `--nosave` |
| `--view-img` | False | Display results | `--view-img` |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | `coco128.yaml` | Dataset config file |
| `--epochs` | `300` | Number of training epochs |
| `--batch-size` | `16` | Batch size (or -1 for auto) |
| `--img-size` | `640` | Training image size |
| `--weights` | `yolov5s.pt` | Starting weights |
| `--device` | `` | CUDA device (e.g., 0 or 0,1,2,3) or cpu |

---

## 🔧 Re-Identification Setup

### Current Status
⚠️ **Important**: The re-ID feature currently uses **dummy embeddings** for demonstration.

### To Make It Production-Ready:

**Step 1**: Prepare your embeddings database
```python
# Create embeddings for your known persons/objects
import torch

# Example: 10 persons, each with a 512-D embedding
real_embeddings = torch.zeros([10, 512])  # Replace with actual embeddings
torch.save(real_embeddings, 'embeddings_db.pt')
```

**Step 2**: Modify `detect.py` (line 66)
```python
# Replace this line:
dum_emb = torch.rand([10, 512]).cuda()

# With this:
dum_emb = torch.load('embeddings_db.pt').cuda()
```

**Step 3**: Remove debug breakpoint (line 207 in detect.py)
```python
# Comment out or remove:
# pdb.set_trace()
```

**Step 4**: Create person ID mapping
```python
# Create a mapping from index to person ID/name
person_names = ['Person1', 'Person2', 'Person3', ...]

# In detection loop, replace print statement with:
most_similar_idx = normalized[0].item()
print(f"Detected: {person_names[most_similar_idx]}")
```

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solutions**:
- Reduce batch size: `--batch-size 8` or `--batch-size 4`
- Reduce image size: `--img-size 320`
- Use smaller model: `--weights yolov5s.pt` instead of `yolov5x.pt`

### Issue: "model_jit.pth not found"
**Solution**: Ensure the re-ID model file is in the repository root directory

### Issue: No detections
**Solutions**:
- Lower confidence threshold: `--conf-thres 0.1`
- Check if correct model is loaded
- Verify image is not corrupted
- Ensure lighting/quality is adequate

### Issue: Too many detections
**Solutions**:
- Increase confidence threshold: `--conf-thres 0.5`
- Adjust IoU threshold: `--iou-thres 0.6`
- Filter by specific classes: `--classes 0`

### Issue: Import errors
**Solution**: 
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Slow inference
**Solutions**:
- Use smaller model (yolov5n or yolov5s)
- Reduce image size: `--img-size 416`
- Enable half-precision: `--half` (requires GPU)
- Use TensorRT export for production

---

## 📊 Understanding Output

### Console Output Example:
```
image 1/1 /path/to/image.jpg: 640x480 2 persons, 1 car, Done. (0.045s)
```
**Meaning**: 
- Image size: 640x480 pixels
- Detected: 2 persons and 1 car
- Inference time: 45ms

### Saved Results:
- **Images**: `runs/detect/exp/*.jpg` (with bounding boxes)
- **Labels**: `runs/detect/exp/labels/*.txt` (if --save-txt used)
- **Crops**: `runs/detect/exp/crops/` (if --save-crop used)

### Label Format (*.txt files):
```
class_id x_center y_center width height confidence
0 0.5 0.5 0.3 0.4 0.85
```
**Note**: Coordinates are normalized (0-1)

---

## 📈 Performance Tips

### For Better Accuracy:
1. Use larger model (yolov5l or yolov5x)
2. Increase image size: `--img-size 1280`
3. Lower confidence threshold: `--conf-thres 0.3`
4. Enable test-time augmentation: `--augment`

### For Faster Inference:
1. Use smaller model (yolov5n or yolov5s)
2. Reduce image size: `--img-size 416`
3. Enable half-precision (GPU): `--half`
4. Export to optimized format (TensorRT, ONNX)

### For Real-time Applications:
```bash
python detect.py --source 0 --weights yolov5s.pt --img-size 416 --half
```

---

## 🎬 Example Workflows

### Workflow 1: Security Camera Analysis
```bash
# Detect persons in surveillance footage
python detect.py --source cctv_footage.mp4 \
    --weights yolov5s.pt \
    --classes 0 \
    --conf-thres 0.4 \
    --save-txt \
    --save-crop
```

### Workflow 2: Traffic Monitoring
```bash
# Detect vehicles (cars, buses, trucks)
python detect.py --source traffic.mp4 \
    --weights yolov5m.pt \
    --classes 2 3 5 7 \
    --save-txt
```

### Workflow 3: Wildlife Camera
```bash
# Detect animals with high confidence
python detect.py --source wildlife/*.jpg \
    --weights yolov5l.pt \
    --conf-thres 0.6 \
    --save-crop
```

---

## 🔗 Resources

- **Full Documentation**: See [CODE_EXPLANATION.md](CODE_EXPLANATION.md)
- **YOLOv5 Docs**: https://docs.ultralytics.com
- **Community**: https://github.com/ultralytics/yolov5/discussions
- **Tutorial**: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

---

## 📝 COCO Class IDs (Common Classes)

| ID | Class | ID | Class | ID | Class |
|----|-------|----|-------|----|-------|
| 0 | person | 1 | bicycle | 2 | car |
| 3 | motorcycle | 5 | bus | 7 | truck |
| 15 | cat | 16 | dog | 17 | horse |
| 24 | backpack | 26 | handbag | 28 | suitcase |
| 39 | bottle | 41 | cup | 56 | chair |
| 57 | couch | 62 | laptop | 63 | mouse |
| 64 | remote | 65 | keyboard | 67 | cell phone |

Full list: https://github.com/ultralytics/yolov5/blob/master/data/coco.yaml

---

## ⚡ Quick Commands Cheat Sheet

```bash
# Basic detection
python detect.py --source image.jpg

# Webcam detection
python detect.py --source 0

# Detect only persons
python detect.py --source video.mp4 --classes 0

# High accuracy mode
python detect.py --source image.jpg --weights yolov5x.pt --img-size 1280

# Fast mode
python detect.py --source video.mp4 --weights yolov5s.pt --img-size 416

# Train custom model
python train.py --data custom.yaml --weights yolov5s.pt --epochs 100

# Validate model
python val.py --weights best.pt --data custom.yaml

# Export to ONNX
python export.py --weights yolov5s.pt --include onnx
```

---

**Need more help?** Check the [CODE_EXPLANATION.md](CODE_EXPLANATION.md) for detailed explanations.
