#!/usr/bin/env python3
"""
Generate Test Images for Autonomous Driving Object Detection
Creates various test scenarios for different detection modes
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as mpatches

def create_synthetic_driving_scene(width=1280, height=384, scene_type="urban"):
    """Create a synthetic driving scene with various objects"""
    
    # Create base image with road
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Sky gradient
    for y in range(height//3):
        intensity = int(135 + (200-135) * y / (height//3))
        img[y, :] = [intensity, intensity, 255]
    
    # Road surface
    road_color = [60, 60, 60]
    img[2*height//3:, :] = road_color
    
    # Road markings
    for x in range(0, width, 100):
        cv2.rectangle(img, (x, height-20), (x+50, height-10), (255, 255, 255), -1)
    
    # Center line
    for x in range(0, width, 80):
        cv2.rectangle(img, (x, height//2 + 20), (x+40, height//2 + 25), (255, 255, 0), -1)
    
    # Buildings/background
    if scene_type == "urban":
        # Add buildings
        building_heights = [random.randint(50, 150) for _ in range(8)]
        building_width = width // 8
        
        for i, h in enumerate(building_heights):
            x1 = i * building_width
            x2 = (i + 1) * building_width
            y1 = height//3 - h
            y2 = 2*height//3
            
            # Building color
            color = [random.randint(80, 120) for _ in range(3)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            
            # Windows
            for wy in range(y1 + 20, y2 - 20, 30):
                for wx in range(x1 + 10, x2 - 10, 20):
                    cv2.rectangle(img, (wx, wy), (wx+8, wy+15), (255, 255, 150), -1)
    
    elif scene_type == "highway":
        # Add trees/vegetation
        for i in range(20):
            x = random.randint(0, width)
            y = random.randint(height//3, 2*height//3)
            cv2.circle(img, (x, y), random.randint(10, 25), (0, 100, 0), -1)
    
    return img

def add_vehicles(img, num_cars=3, include_small=True):
    """Add vehicle objects to the scene"""
    height, width = img.shape[:2]
    objects = []
    
    for i in range(num_cars):
        # Vehicle size and position
        if include_small and i < num_cars//2:
            # Small distant vehicles
            car_width = random.randint(30, 60)
            car_height = random.randint(20, 35)
            x = random.randint(50, width - car_width - 50)
            y = random.randint(height//2, height//2 + 50)
        else:
            # Larger closer vehicles
            car_width = random.randint(80, 120)
            car_height = random.randint(45, 70)
            x = random.randint(50, width - car_width - 50)
            y = random.randint(height//2 + 40, height - car_height - 30)
        
        # Vehicle color
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (128, 128, 128)]
        color = random.choice(colors)
        
        # Draw vehicle body
        cv2.rectangle(img, (x, y), (x + car_width, y + car_height), color, -1)
        
        # Add details
        # Windows
        cv2.rectangle(img, (x+5, y+5), (x + car_width-5, y + car_height//2), (100, 150, 255), -1)
        
        # Wheels
        wheel_radius = car_height // 6
        cv2.circle(img, (x + car_width//4, y + car_height), wheel_radius, (0, 0, 0), -1)
        cv2.circle(img, (x + 3*car_width//4, y + car_height), wheel_radius, (0, 0, 0), -1)
        
        objects.append({
            'class': 'car',
            'bbox': [x, y, x + car_width, y + car_height],
            'size': 'small' if car_width < 70 else 'large'
        })
    
    return img, objects

def add_pedestrians(img, num_pedestrians=2):
    """Add pedestrian objects to the scene"""
    height, width = img.shape[:2]
    objects = []
    
    for i in range(num_pedestrians):
        # Pedestrian size and position
        ped_width = random.randint(15, 25)
        ped_height = random.randint(40, 60)
        x = random.randint(50, width - ped_width - 50)
        y = random.randint(height//2 + 60, height - ped_height - 10)
        
        # Draw pedestrian (simplified)
        # Body
        cv2.rectangle(img, (x, y + ped_height//3), (x + ped_width, y + ped_height), (100, 50, 150), -1)
        
        # Head
        cv2.circle(img, (x + ped_width//2, y + ped_height//6), ped_width//3, (200, 150, 120), -1)
        
        objects.append({
            'class': 'pedestrian',
            'bbox': [x, y, x + ped_width, y + ped_height],
            'size': 'medium'
        })
    
    return img, objects

def add_traffic_signs(img, num_signs=2):
    """Add traffic sign objects"""
    height, width = img.shape[:2]
    objects = []
    
    for i in range(num_signs):
        # Sign size and position
        sign_size = random.randint(20, 40)
        x = random.randint(50, width - sign_size - 50)
        y = random.randint(height//3, height//2 + 20)
        
        # Sign colors
        sign_types = [
            ((255, 255, 255), (255, 0, 0)),  # Stop sign
            ((255, 255, 0), (0, 0, 0)),      # Warning
            ((0, 255, 0), (255, 255, 255))   # Go
        ]
        bg_color, fg_color = random.choice(sign_types)
        
        # Draw sign
        cv2.rectangle(img, (x, y), (x + sign_size, y + sign_size), bg_color, -1)
        cv2.rectangle(img, (x+2, y+2), (x + sign_size-2, y + sign_size-2), fg_color, 2)
        
        objects.append({
            'class': 'traffic_sign',
            'bbox': [x, y, x + sign_size, y + sign_size],
            'size': 'small'
        })
    
    return img, objects

def create_test_scenarios():
    """Create different test scenarios for various detection modes"""
    
    scenarios = {
        'urban_dense': {
            'description': 'Dense urban scene with many small objects',
            'scene_type': 'urban',
            'cars': 5,
            'pedestrians': 3,
            'signs': 4,
            'include_small': True
        },
        'highway_sparse': {
            'description': 'Highway scene with sparse larger objects',
            'scene_type': 'highway',
            'cars': 2,
            'pedestrians': 0,
            'signs': 1,
            'include_small': False
        },
        'mixed_objects': {
            'description': 'Mixed scene for comprehensive testing',
            'scene_type': 'urban',
            'cars': 4,
            'pedestrians': 2,
            'signs': 3,
            'include_small': True
        },
        'small_objects_challenge': {
            'description': 'Scene designed to test small object detection',
            'scene_type': 'urban',
            'cars': 6,
            'pedestrians': 4,
            'signs': 6,
            'include_small': True
        },
        'domain_adaptation_sim': {
            'description': 'Simulation-style scene for domain adaptation testing',
            'scene_type': 'highway',
            'cars': 3,
            'pedestrians': 1,
            'signs': 2,
            'include_small': True
        }
    }
    
    return scenarios

def save_scene_info(objects, scenario_name, description):
    """Save ground truth information for each scene"""
    info = {
        'scenario': scenario_name,
        'description': description,
        'objects': objects,
        'stats': {
            'total_objects': len(objects),
            'cars': len([o for o in objects if o['class'] == 'car']),
            'pedestrians': len([o for o in objects if o['class'] == 'pedestrian']),
            'signs': len([o for o in objects if o['class'] == 'traffic_sign']),
            'small_objects': len([o for o in objects if o['size'] == 'small'])
        }
    }
    
    with open(f'test_images/{scenario_name}_info.txt', 'w') as f:
        f.write(f"Scenario: {info['scenario']}\n")
        f.write(f"Description: {info['description']}\n")
        f.write(f"Total Objects: {info['stats']['total_objects']}\n")
        f.write(f"Cars: {info['stats']['cars']}\n")
        f.write(f"Pedestrians: {info['stats']['pedestrians']}\n")
        f.write(f"Traffic Signs: {info['stats']['signs']}\n")
        f.write(f"Small Objects: {info['stats']['small_objects']}\n\n")
        
        f.write("Object Details:\n")
        for i, obj in enumerate(objects):
            f.write(f"{i+1}. {obj['class']} ({obj['size']}) - BBox: {obj['bbox']}\n")

def create_real_world_style_images():
    """Create more realistic test images"""
    
    # KITTI-style image
    kitti_img = np.zeros((375, 1242, 3), dtype=np.uint8)
    
    # Sky
    kitti_img[:150, :] = [180, 200, 255]
    
    # Buildings/background
    kitti_img[150:250, :] = [120, 120, 120]
    
    # Road
    kitti_img[250:, :] = [80, 80, 80]
    
    # Add lane markings
    for x in range(0, 1242, 60):
        cv2.rectangle(kitti_img, (x, 350), (x+30, 355), (255, 255, 255), -1)
    
    # Add realistic vehicles
    vehicles = [
        {'x': 400, 'y': 280, 'w': 120, 'h': 60, 'color': (0, 0, 180)},
        {'x': 600, 'y': 290, 'w': 100, 'h': 50, 'color': (150, 150, 150)},
        {'x': 800, 'y': 295, 'w': 80, 'h': 40, 'color': (255, 255, 255)},
        {'x': 1000, 'y': 300, 'w': 60, 'h': 30, 'color': (0, 0, 255)},  # Small distant car
    ]
    
    for v in vehicles:
        cv2.rectangle(kitti_img, (v['x'], v['y']), (v['x']+v['w'], v['y']+v['h']), v['color'], -1)
        # Add windows
        cv2.rectangle(kitti_img, (v['x']+5, v['y']+5), (v['x']+v['w']-5, v['y']+v['h']//2), (100, 150, 200), -1)
    
    cv2.imwrite('test_images/kitti_style_scene.jpg', kitti_img)
    
    # CARLA-style image (more saturated, perfect lighting)
    carla_img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Bright sky
    carla_img[:200, :] = [200, 220, 255]
    
    # Clean buildings
    for i in range(5):
        x1 = i * 160
        x2 = (i + 1) * 160
        height = 100 + i * 20
        cv2.rectangle(carla_img, (x1, 200), (x2, 200 + height), (100 + i*20, 100 + i*20, 120), -1)
    
    # Perfect road
    carla_img[400:, :] = [60, 60, 60]
    
    # Clean lane markings
    for x in range(0, 800, 80):
        cv2.rectangle(carla_img, (x, 580), (x+40, 585), (255, 255, 255), -1)
    
    # Add perfect vehicles
    perfect_vehicles = [
        {'x': 200, 'y': 450, 'w': 100, 'h': 50, 'color': (255, 0, 0)},
        {'x': 400, 'y': 460, 'w': 90, 'h': 45, 'color': (0, 255, 0)},
        {'x': 600, 'y': 470, 'w': 70, 'h': 35, 'color': (0, 0, 255)},
    ]
    
    for v in perfect_vehicles:
        cv2.rectangle(carla_img, (v['x'], v['y']), (v['x']+v['w'], v['y']+v['h']), v['color'], -1)
        cv2.rectangle(carla_img, (v['x']+5, v['y']+5), (v['x']+v['w']-5, v['y']+v['h']//2), (150, 200, 255), -1)
    
    cv2.imwrite('test_images/carla_style_scene.jpg', carla_img)

def main():
    """Generate all test images"""
    
    print("🖼️  Generating Test Images for Object Detection...")
    print("=" * 60)
    
    # Ensure test_images directory exists
    os.makedirs('test_images', exist_ok=True)
    
    scenarios = create_test_scenarios()
    
    total_objects_generated = 0
    
    for scenario_name, config in scenarios.items():
        print(f"\n📸 Creating {scenario_name}...")
        print(f"   Description: {config['description']}")
        
        # Create base scene
        img = create_synthetic_driving_scene(
            scene_type=config['scene_type']
        )
        
        all_objects = []
        
        # Add vehicles
        img, car_objects = add_vehicles(
            img, 
            num_cars=config['cars'],
            include_small=config['include_small']
        )
        all_objects.extend(car_objects)
        
        # Add pedestrians
        img, ped_objects = add_pedestrians(
            img, 
            num_pedestrians=config['pedestrians']
        )
        all_objects.extend(ped_objects)
        
        # Add traffic signs
        img, sign_objects = add_traffic_signs(
            img, 
            num_signs=config['signs']
        )
        all_objects.extend(sign_objects)
        
        # Save image
        cv2.imwrite(f'test_images/{scenario_name}.jpg', img)
        
        # Save scene information
        save_scene_info(all_objects, scenario_name, config['description'])
        
        total_objects_generated += len(all_objects)
        
        print(f"   ✓ Generated {len(all_objects)} objects")
        print(f"   ✓ Small objects: {len([o for o in all_objects if o['size'] == 'small'])}")
    
    # Create realistic style images
    print(f"\n📸 Creating realistic style reference images...")
    create_real_world_style_images()
    
    # Create a summary
    print(f"\n📊 Generation Summary:")
    print(f"   ✓ Total scenarios: {len(scenarios)}")
    print(f"   ✓ Total objects: {total_objects_generated}")
    print(f"   ✓ Additional reference images: 2")
    print(f"   ✓ All images saved to: test_images/")
    
    # Create index file
    with open('test_images/README.md', 'w') as f:
        f.write("# Test Images for Object Detection\n\n")
        f.write("This directory contains synthetic test images for various object detection scenarios.\n\n")
        f.write("## Test Scenarios\n\n")
        
        for scenario_name, config in scenarios.items():
            f.write(f"### {scenario_name}.jpg\n")
            f.write(f"- **Description**: {config['description']}\n")
            f.write(f"- **Scene Type**: {config['scene_type']}\n")
            f.write(f"- **Objects**: {config['cars']} cars, {config['pedestrians']} pedestrians, {config['signs']} signs\n")
            f.write(f"- **Info File**: {scenario_name}_info.txt\n\n")
        
        f.write("## Reference Images\n\n")
        f.write("- **kitti_style_scene.jpg**: KITTI dataset style image\n")
        f.write("- **carla_style_scene.jpg**: CARLA simulation style image\n\n")
        
        f.write("## Usage\n\n")
        f.write("These images can be used to test:\n")
        f.write("- Standard object detection\n")
        f.write("- Parallel patch detection (especially for small objects)\n")
        f.write("- Domain adaptation (comparing CARLA vs KITTI styles)\n")
        f.write("- Unsupervised detection algorithms\n")
    
    print(f"\n🎉 Test image generation complete!")
    print(f"📁 Check the test_images/ directory for all generated files.")

if __name__ == "__main__":
    main() 