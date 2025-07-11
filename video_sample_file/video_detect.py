#!/usr/bin/env python3
"""
Enhanced Video Object Detection with Full Project Integration
Supports all project functionality: object detection, patch detection, domain adaptation, tracking
"""

import cv2
import numpy as np
import torch
import yaml
import argparse
import time
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import requests

# Add project src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

# Try to import main project models
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available")

try:
    from models.object_detection import YOLOv8Detector, ParallelPatchDetector
    from models.domain_adaptation import DomainAdversarialNetwork
    from models.tracking import DeepSORT
    from data.transforms import PatchExtractor
    from unsupervised.lost import LOSTDetector
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("Warning: Main project models not available, using simplified detection")


class VideoProcessor:
    """Enhanced video processor with full project integration"""
    
    def __init__(self, config_path: str = None):
        """Initialize video processor with configuration"""
        self.config = self._load_config(config_path)
        self.models = {}
        self.tracking_enabled = False
        self.api_url = self.config.get('api_url', 'http://localhost:8000')
        self.use_api = self.config.get('use_api', False)
        
        # Initialize models
        self._initialize_models()
        
        # Video processing stats
        self.frame_count = 0
        self.detection_count = 0
        self.processing_times = []
        
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from YAML file or use defaults"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'models': {
                'object_detection': {
                    'architecture': 'yolov8',
                    'model_path': 'yolov8n.pt',
                    'confidence_threshold': 0.5,
                    'nms_threshold': 0.4,
                    'num_classes': 80
                },
                'tracking': {
                    'enabled': True,
                    'max_age': 30,
                    'min_hits': 3
                },
                'domain_adaptation': {
                    'enabled': False,
                    'lambda_grl': 1.0
                }
            },
            'inference': {
                'patch_detection': {
                    'enabled': True,
                    'patch_size': [192, 192],
                    'overlap': 0.2,
                    'min_object_size': 20
                },
                'parallel_processing': {
                    'batch_size': 4,
                    'num_workers': 2
                }
            },
            'video': {
                'output_format': 'mp4v',
                'visualization': {
                    'show_confidence': True,
                    'show_class_names': True,
                    'show_tracking_ids': True,
                    'bbox_thickness': 2,
                    'font_scale': 0.6
                }
            },
            'api_url': 'http://localhost:8000',
            'use_api': False
        }
    
    def _initialize_models(self):
        """Initialize detection and tracking models"""
        print("🔧 Initializing models...")
        
        # Object Detection Model
        if MODELS_AVAILABLE:
            try:
                self.models['yolov8'] = YOLOv8Detector(
                    num_classes=self.config['models']['object_detection']['num_classes'],
                    confidence_threshold=self.config['models']['object_detection']['confidence_threshold'],
                    nms_threshold=self.config['models']['object_detection']['nms_threshold']
                )
                
                # Patch detector
                if self.config['inference']['patch_detection']['enabled']:
                    self.models['patch_detector'] = ParallelPatchDetector(
                        base_detector=self.models['yolov8'],
                        patch_size=tuple(self.config['inference']['patch_detection']['patch_size']),
                        overlap=self.config['inference']['patch_detection']['overlap'],
                        min_object_size=self.config['inference']['patch_detection']['min_object_size']
                    )
                
                # Domain adaptation
                if self.config['models']['domain_adaptation']['enabled']:
                    self.models['domain_adaptation'] = DomainAdversarialNetwork(
                        num_classes=self.config['models']['object_detection']['num_classes'],
                        lambda_grl=self.config['models']['domain_adaptation']['lambda_grl']
                    )
                
                # Tracking
                if self.config['models']['tracking']['enabled']:
                    self.models['tracker'] = DeepSORT(
                        max_age=self.config['models']['tracking']['max_age'],
                        min_hits=self.config['models']['tracking']['min_hits']
                    )
                    self.tracking_enabled = True
                
                print("✅ Main project models loaded successfully")
                
            except Exception as e:
                print(f"⚠️ Error loading main project models: {e}")
                self._fallback_to_simple_yolo()
        else:
            self._fallback_to_simple_yolo()
    
    def _fallback_to_simple_yolo(self):
        """Fallback to simple YOLO model"""
        if YOLO_AVAILABLE:
            try:
                model_path = self.config['models']['object_detection']['model_path']
                if os.path.exists(model_path):
                    self.models['simple_yolo'] = YOLO(model_path)
                    print(f"✅ Loaded simple YOLO from {model_path}")
                else:
                    print(f"❌ Model file not found: {model_path}")
            except Exception as e:
                print(f"❌ Error loading YOLO: {e}")
    
    def detect_objects_api(self, frame: np.ndarray) -> List[Dict]:
        """Use API for object detection"""
        try:
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to bytes
            import io
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            # API request
            files = {"file": img_bytes.getvalue()}
            params = {
                "use_patch_detection": str(self.config['inference']['patch_detection']['enabled']).lower(),
                "confidence_threshold": self.config['models']['object_detection']['confidence_threshold'],
                "nms_threshold": self.config['models']['object_detection']['nms_threshold']
            }
            
            response = requests.post(
                f"{self.api_url}/detect",
                files=files,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('detections', [])
            else:
                print(f"API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"API detection error: {e}")
            return []
    
    def detect_objects_local(self, frame: np.ndarray) -> List[Dict]:
        """Local object detection using loaded models"""
        detections = []
        
        try:
            # Convert frame to tensor if using main project models
            if 'yolov8' in self.models or 'patch_detector' in self.models:
                # Use main project models
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Convert to tensor (simplified preprocessing)
                frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                
                # Select detection model
                if self.config['inference']['patch_detection']['enabled'] and 'patch_detector' in self.models:
                    results = self.models['patch_detector'](frame_tensor)
                elif 'yolov8' in self.models:
                    results = self.models['yolov8'].predict(frame_tensor)
                else:
                    return []
                
                # Convert to standard format
                if isinstance(results, list) and len(results) > 0:
                    result = results[0]
                    detections = self._convert_detections_to_standard(result, frame.shape)
                
            elif 'simple_yolo' in self.models:
                # Use simple YOLO
                results = self.models['simple_yolo'](frame)
                detections = self._convert_yolo_results(results, frame.shape)
            
        except Exception as e:
            print(f"Local detection error: {e}")
        
        return detections
    
    def _convert_detections_to_standard(self, detections: Dict, frame_shape: Tuple[int, int, int]) -> List[Dict]:
        """Convert model outputs to standard detection format"""
        standard_detections = []
        
        try:
            # Handle different detection formats
            if 'boxes' in detections and 'scores' in detections:
                boxes = detections['boxes']
                scores = detections['scores']
                classes = detections.get('classes', [0] * len(boxes))
                
                for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                    if score >= self.config['models']['object_detection']['confidence_threshold']:
                        detection = {
                            'bbox': [float(x) for x in box[:4]],  # [x1, y1, x2, y2]
                            'confidence': float(score),
                            'class_id': int(cls),
                            'class_name': self._get_class_name(int(cls)),
                            'tracking_id': None
                        }
                        standard_detections.append(detection)
            
        except Exception as e:
            print(f"Error converting detections: {e}")
        
        return standard_detections
    
    def _convert_yolo_results(self, results, frame_shape: Tuple[int, int, int]) -> List[Dict]:
        """Convert YOLO results to standard format"""
        detections = []
        
        try:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Get box coordinates
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = boxes.conf[i].cpu().numpy()
                        cls = int(boxes.cls[i].cpu().numpy())
                        
                        if conf >= self.config['models']['object_detection']['confidence_threshold']:
                            detection = {
                                'bbox': [float(x) for x in box],
                                'confidence': float(conf),
                                'class_id': cls,
                                'class_name': self._get_class_name(cls),
                                'tracking_id': None
                            }
                            detections.append(detection)
        
        except Exception as e:
            print(f"Error converting YOLO results: {e}")
        
        return detections
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        # COCO class names
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        if 0 <= class_id < len(coco_classes):
            return coco_classes[class_id]
        return f"class_{class_id}"
    
    def update_tracking(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Update object tracking"""
        if not self.tracking_enabled or 'tracker' not in self.models:
            return detections
        
        try:
            # Convert detections to tracking format
            det_boxes = []
            det_scores = []
            
            for det in detections:
                bbox = det['bbox']
                det_boxes.append([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])  # [x, y, w, h]
                det_scores.append(det['confidence'])
            
            if det_boxes:
                det_boxes = np.array(det_boxes)
                det_scores = np.array(det_scores)
                
                # Update tracker
                tracks = self.models['tracker'].update(det_boxes, det_scores, frame)
                
                # Assign tracking IDs
                for i, track in enumerate(tracks):
                    if i < len(detections):
                        detections[i]['tracking_id'] = int(track[4])  # tracking ID
        
        except Exception as e:
            print(f"Tracking error: {e}")
        
        return detections
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        viz_config = self.config['video']['visualization']
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            tracking_id = detection.get('tracking_id')
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Color based on class or tracking ID
            if tracking_id is not None:
                color = self._get_tracking_color(tracking_id)
            else:
                color = (0, 255, 0)  # Green
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, viz_config['bbox_thickness'])
            
            # Prepare label
            label_parts = []
            if viz_config['show_class_names']:
                label_parts.append(class_name)
            if viz_config['show_confidence']:
                label_parts.append(f"{confidence:.2f}")
            if viz_config['show_tracking_ids'] and tracking_id is not None:
                label_parts.append(f"ID:{tracking_id}")
            
            label = " ".join(label_parts)
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, viz_config['font_scale'], 1
            )
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, viz_config['font_scale'],
                (255, 255, 255), 1
            )
        
        return frame
    
    def _get_tracking_color(self, tracking_id: int) -> Tuple[int, int, int]:
        """Get consistent color for tracking ID"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (128, 128, 128), (255, 99, 71), (60, 179, 113)
        ]
        return colors[tracking_id % len(colors)]
    
    def process_video(self, input_path: str, output_path: str = None):
        """Process video with object detection and tracking"""
        if not os.path.exists(input_path):
            print(f"❌ Input video not found: {input_path}")
            return
        
        if output_path is None:
            output_path = input_path.replace('.', '_processed.')
        
        print(f"🎬 Processing video: {input_path}")
        print(f"📤 Output will be saved to: {output_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("❌ Error: Cannot open input video.")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video info: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*self.config['video']['output_format'])
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing loop
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # Object detection
            if self.use_api:
                detections = self.detect_objects_api(frame)
            else:
                detections = self.detect_objects_local(frame)
            
            # Update tracking
            detections = self.update_tracking(detections, frame)
            
            # Visualize detections
            annotated_frame = self.visualize_detections(frame, detections)
            
            # Write frame
            out.write(annotated_frame)
            
            # Update stats
            self.frame_count += 1
            self.detection_count += len(detections)
            frame_time = time.time() - frame_start
            self.processing_times.append(frame_time)
            
            # Progress indicator
            if self.frame_count % 30 == 0:  # Every second at 30fps
                progress = (self.frame_count / total_frames) * 100
                avg_time = np.mean(self.processing_times[-30:])
                print(f"⏳ Progress: {progress:.1f}% | Frame {self.frame_count}/{total_frames} | "
                      f"Avg time: {avg_time:.3f}s/frame | Detections: {len(detections)}")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Processing summary
        total_time = time.time() - start_time
        avg_processing_time = np.mean(self.processing_times)
        print(f"\n✅ Video processing completed!")
        print(f"📊 Processing Summary:")
        print(f"   - Total frames: {self.frame_count}")
        print(f"   - Total detections: {self.detection_count}")
        print(f"   - Processing time: {total_time:.2f}s")
        print(f"   - Average time per frame: {avg_processing_time:.3f}s")
        print(f"   - Average FPS: {1/avg_processing_time:.1f}")
        print(f"   - Output saved to: {output_path}")
    
    def process_real_time(self, source: int = 0):
        """Process real-time video from camera"""
        print(f"📹 Starting real-time processing from camera {source}")
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open camera {source}")
            return
        
        print("🎮 Real-time detection active. Press 'q' to quit, 's' to save frame")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading from camera")
                break
            
            start_time = time.time()
            
            # Object detection
            if self.use_api:
                detections = self.detect_objects_api(frame)
            else:
                detections = self.detect_objects_local(frame)
            
            # Update tracking
            detections = self.update_tracking(detections, frame)
            
            # Visualize detections
            annotated_frame = self.visualize_detections(frame, detections)
            
            # Add info overlay
            processing_time = time.time() - start_time
            fps = 1 / processing_time if processing_time > 0 else 0
            
            info_text = f"FPS: {fps:.1f} | Detections: {len(detections)} | Frame: {frame_count}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Real-time Object Detection", annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"frame_{frame_count:04d}.jpg"
                cv2.imwrite(save_path, annotated_frame)
                print(f"📸 Frame saved to: {save_path}")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Real-time processing stopped")


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced Video Object Detection")
    parser.add_argument("--input", "-i", type=str, help="Input video path")
    parser.add_argument("--output", "-o", type=str, help="Output video path")
    parser.add_argument("--config", "-c", type=str, help="Configuration YAML file")
    parser.add_argument("--realtime", "-r", action="store_true", help="Real-time camera processing")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--api", action="store_true", help="Use API for detection")
    parser.add_argument("--patch-detection", action="store_true", help="Enable patch detection")
    parser.add_argument("--tracking", action="store_true", help="Enable object tracking")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = VideoProcessor(args.config)
    
    # Override config with command line arguments
    if args.api:
        processor.use_api = True
    if args.patch_detection:
        processor.config['inference']['patch_detection']['enabled'] = True
    if args.tracking:
        processor.config['models']['tracking']['enabled'] = True
    if args.confidence:
        processor.config['models']['object_detection']['confidence_threshold'] = args.confidence
    
    # Process video or real-time
    if args.realtime:
        processor.process_real_time(args.camera)
    elif args.input:
        processor.process_video(args.input, args.output)
    else:
        print("❌ Please specify --input for video file or --realtime for camera")
        print("💡 Examples:")
        print("   python video_detect.py --input video.mp4 --output result.mp4")
        print("   python video_detect.py --realtime --camera 0")
        print("   python video_detect.py --input video.mp4 --api --patch-detection --tracking")


if __name__ == "__main__":
    main()