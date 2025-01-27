"""
Real-time object detection using YOLOv8 with MPS acceleration.
Author: Kushvinth Madhavan
GitHub: https://github.com/Kushvinth-Madhavan
"""

from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import torch

class ObjectDetector:
    def __init__(self):
        """Initialize the ObjectDetector with device configuration and model setup."""
        # Configure device (MPS for Apple Silicon, CPU as fallback)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        
        # Initialize video capture
        self.setup_video_capture()
        
        # Load YOLO model
        self.model = self.load_model()
        
        # Initialize FPS calculation variables
        self.prev_frame_time = 0
        self.new_frame_time = 0
        
        # Load class names
        self.class_names = self.load_class_names()

    def setup_video_capture(self):
        """Setup video capture with specified resolution."""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 384)  # Width
        self.cap.set(4, 640)  # Height
        
        if not self.cap.isOpened():
            raise Exception("Error: Could not open video capture device")

    def load_model(self):
        """Load and configure YOLO model."""
        try:
            model = YOLO("../Yolo-Weights/yolov8n.pt")
            model.to(self.device)
            return model
        except Exception as e:
            raise Exception(f"Error loading YOLO model: {str(e)}")

    @staticmethod
    def load_class_names():
        """Load class names for object detection."""
        return ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    def process_frame(self, img):
        """Process a single frame for object detection."""
        try:
            results = self.model(img, stream=True, device=self.device)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Extract coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    
                    # Draw bounding box
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    
                    # Get confidence and class
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    
                    # Add label
                    cvzone.putTextRect(img, f'{self.class_names[cls]} {conf}',
                                     (max(0, x1), max(35, y1)),
                                     scale=1, thickness=1)
            
            return img
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return img

    def calculate_fps(self):
        """Calculate and return the current FPS."""
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        return fps

    def run(self):
        """Main detection loop."""
        try:
            while True:
                success, img = self.cap.read()
                if not success:
                    print("Error: Failed to read frame")
                    break

                # Process frame
                img = self.process_frame(img)
                
                # Calculate and display FPS
                fps = self.calculate_fps()
                print(f"FPS: {fps:.2f}")
                
                # Display the frame
                cv2.imshow("YOLOv8 Detection", img)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Detection stopped by user")
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = ObjectDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {str(e)}")
