# YOLOv8 Real-Time Object Detection with MPS

A real-time object detection system using YOLOv8 with Metal Performance Shaders (MPS) acceleration for Apple Silicon. This project leverages the power of M1/M2 chips for efficient object detection through a connected camera feed.

## 🚀 Features

- Real-time object detection with YOLOv8
- MPS acceleration for Apple Silicon
- Support for 80+ object classes
- Live FPS monitoring
- Bounding box visualization
- Confidence score display

## 💻 Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.8+
- External camera/webcam
- Git

## 🛠️ Installation

1. Clone the repository
```bash
git clone https://github.com/Kushvinth-Madhavan/yolov8-realtime-detection.git
cd yolov8-realtime-detection
```

2. Set up virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download YOLOv8 weights
```bash
mkdir -p Yolo-Weights
# Download yolov8n.pt to Yolo-Weights directory
```

## 🎯 Usage

Run the detection script:
```bash
python src/detect.py
```

Exit the application by pressing 'q'.

## 📁 Project Structure
```
yolov8-realtime-detection/
├── src/
│   ├── __init__.py
│   └── detect.py
├── Yolo-Weights/
│   └── yolov8n.pt
├── demo/
│   └── demo.png
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Configuration

Default settings:
- Resolution: 384x640
- Model: YOLOv8n
- Acceleration: MPS (if available)
- Classes: 80+ common objects

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👨‍💻 Author

Kushvinth Madhavan
- GitHub: [@Kushvinth-Madhavan](https://github.com/Kushvinth-Madhavan)

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [cvzone](https://github.com/cvzone/cvzone)
