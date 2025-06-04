# generate_sem_dataset_azure_cldai.py
# Enhanced SEM Dataset Generator optimized for Azure ML pipeline integration

import traceback
import os
import csv
import random
import argparse
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from scipy import ndimage
import logging
import json
from pathlib import Path

# Azure ML imports
try:
    from azureml.core import Run, Dataset
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_fractal_roughness(height, roughness_strength=0.8, frequency_scale=0.15):
    """Generate subtle but visible fractal-like roughness profile"""
    # Create base roughness - increased for more visible edges
    base_freq = np.random.normal(0, roughness_strength * 0.7, size=height)
    
    # Add fine details
    fine_noise = np.random.normal(0, roughness_strength * 0.4, size=height)
    
    # Add gentle undulations
    coarse_variation = np.sin(np.linspace(0, 3*np.pi * frequency_scale, height)) * roughness_strength * 0.6
    
    # Combine all components
    combined_profile = base_freq + fine_noise + coarse_variation
    
    # Apply moderate smoothing
    kernel_size = max(3, int(height * 0.03))
    kernel = np.ones(kernel_size) / kernel_size
    smooth_profile = np.convolve(combined_profile, kernel, mode='same')
    
    # Add occasional small spikes for more realistic appearance
    if height > 40:
        spike_positions = np.random.choice(height, size=max(1, height//40), replace=False)
        for pos in spike_positions:
            smooth_profile[pos] += np.random.normal(0, roughness_strength * 0.8)
    
    # Clamp to reasonable but more visible range
    smooth_profile = np.clip(smooth_profile, -3, 3)
    return smooth_profile.astype(int)

def generate_edge_variation(height, base_variation=0.5):
    """Generate more noticeable edge variations that simulate real SEM line irregularities"""
    # Create more visible Perlin-like noise
    variation = np.zeros(height)
    frequency = 1.0
    amplitude = base_variation
    
    for octave in range(3):  # More octaves for better variation
        octave_variation = np.sin(np.linspace(0, frequency * 1.5 * np.pi, height))
        octave_variation += np.random.normal(0, 0.15, height)  # Slightly more randomness
        variation += octave_variation * amplitude
        
        frequency *= 1.8
        amplitude *= 0.6
    
    # Add occasional small local variations
    if height > 25:
        for _ in range(max(1, height//45)):
            pos = random.randint(0, height-1)
            variation[pos:pos+2] += np.random.normal(0, base_variation * 0.8, min(2, height-pos))
    
    # Clamp to more visible range
    variation = np.clip(variation, -2, 2)
    return variation.astype(int)

def draw_realistic_rough_line(draw, x1, y1, x2, y2, fill, line_width):
    """Draw a line with more visible SEM-like roughness on both edges"""
    height = y2 - y1
    
    # Generate more noticeable roughness for left and right edges
    left_roughness = generate_fractal_roughness(height, roughness_strength=0.6)
    right_roughness = generate_fractal_roughness(height, roughness_strength=0.6)
    
    # Add more visible edge variations
    left_edge_var = generate_edge_variation(height, base_variation=0.4)
    right_edge_var = generate_edge_variation(height, base_variation=0.4)
    
    # Combine roughness and edge variations
    left_edge = left_roughness + left_edge_var
    right_edge = right_roughness + right_edge_var
    
    # Draw the line with more visible variations
    for i in range(height):
        y = y1 + i
        if y >= y2:
            break
            
        # Calculate edge positions with bounds checking
        left_x = max(x1, x1 + left_edge[i])
        right_x = min(x2, x2 + right_edge[i])
        
        # Ensure minimum line width is maintained (reduced threshold for more variation)
        actual_width = right_x - left_x
        if actual_width < line_width * 0.5:  # Allow more variation (was 0.7)
            center = (left_x + right_x) / 2
            left_x = max(x1, center - line_width * 0.25)
            right_x = min(x2, center + line_width * 0.25)
        
        # Draw the line segment
        if right_x > left_x:
            # Add subtle internal variations
            segment_fill = fill + np.random.normal(0, 4)  # Slightly more variation
            segment_fill = max(0, min(255, int(segment_fill)))
            draw.line([(int(left_x), y), (int(right_x), y)], fill=segment_fill)

def add_surface_texture(img_array, texture_strength=5):
    """Add subtle surface texture to simulate SEM imaging artifacts"""
    height, width = img_array.shape
    
    # Generate very subtle Perlin-like noise for surface texture
    x_coords = np.linspace(0, 2, width)  # Reduced frequency
    y_coords = np.linspace(0, 2, height)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Much more subtle frequency components
    texture = np.sin(X * np.pi) * np.cos(Y * np.pi) * texture_strength * 0.1
    texture += np.sin(X * 3 * np.pi) * np.cos(Y * 3 * np.pi) * texture_strength * 0.05
    texture += np.random.normal(0, texture_strength * 0.2, (height, width))
    
    # Apply texture
    textured = img_array.astype(np.float32) + texture
    return np.clip(textured, 0, 255).astype(np.uint8)

def generate_organic_shape(width, height, shape_type="gap"):
    """Generate organic, curved shape for realistic gaps and bridges"""
    # Create a mask for the organic shape
    mask = np.zeros((height, width), dtype=bool)
    
    if shape_type == "gap":
        # For gaps, create an elliptical/oval shape with irregular edges
        center_y = height // 2
        center_x = width // 2
        
        # Base ellipse parameters
        a = width * 0.4  # Semi-major axis
        b = height * 0.3  # Semi-minor axis
        
        for y in range(height):
            for x in range(width):
                # Distance from center
                dx = (x - center_x) / a
                dy = (y - center_y) / b
                
                # Ellipse equation with noise for organic shape
                noise = np.random.normal(0, 0.1)
                ellipse_value = dx*dx + dy*dy + noise
                
                if ellipse_value < 0.8:  # Inside the ellipse with some variation
                    mask[y, x] = True
                    
        # Add some irregular extensions
        for _ in range(random.randint(2, 4)):
            ext_y = random.randint(0, height-1)
            ext_x = random.randint(0, width-1)
            ext_size = random.randint(2, 4)
            
            for dy in range(-ext_size, ext_size+1):
                for dx in range(-ext_size, ext_size+1):
                    ny, nx = ext_y + dy, ext_x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if dx*dx + dy*dy <= ext_size*ext_size:
                            mask[ny, nx] = True
    
    elif shape_type == "bridge":
        # For bridges, create a more elongated, curved connection
        # Start from one side and curve to the other
        start_y = random.randint(height//4, 3*height//4)
        end_y = start_y + random.randint(-height//3, height//3)
        
        # Create curved path using quadratic bezier
        for t in np.linspace(0, 1, width):
            # Quadratic bezier curve
            control_y = (start_y + end_y) / 2 + random.randint(-height//4, height//4)
            
            # Calculate point on curve
            curve_y = int((1-t)**2 * start_y + 2*(1-t)*t * control_y + t**2 * end_y)
            curve_x = int(t * width)
            
            # Add thickness around the curve
            thickness = random.randint(1, 3)
            for dy in range(-thickness, thickness+1):
                for dx in range(-thickness//2, thickness//2+1):
                    ny, nx = curve_y + dy, curve_x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        mask[ny, nx] = True
        
        # Add some bulging areas to make it more organic
        for _ in range(random.randint(1, 3)):
            bulge_y = random.randint(0, height-1)
            bulge_x = random.randint(0, width-1)
            bulge_size = random.randint(2, 4)
            
            for dy in range(-bulge_size, bulge_size+1):
                for dx in range(-bulge_size, bulge_size+1):
                    ny, nx = bulge_y + dy, bulge_x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if dx*dx + dy*dy <= bulge_size*bulge_size * 0.7:
                            mask[ny, nx] = True
    
    return mask

def draw_organic_gap(draw, x1, x2, center_y, height, fill):
    """Draw organic, curved gap similar to real SEM images"""
    width = x2 - x1
    gap_height = height
    
    # Generate organic shape
    organic_mask = generate_organic_shape(width, gap_height, "gap")
    
    # Apply the organic shape
    start_y = center_y - gap_height // 2
    for y in range(gap_height):
        for x in range(width):
            if organic_mask[y, x]:
                actual_x = x1 + x
                actual_y = start_y + y
                # Add some edge softening
                edge_intensity = fill + np.random.normal(0, 5)
                edge_intensity = max(0, min(255, int(edge_intensity)))
                try:
                    draw.point((actual_x, actual_y), fill=edge_intensity)
                except:
                    continue

def draw_organic_bridge(draw, x1, x2, y, height, fill):
    """Draw organic, curved bridge similar to real SEM images"""
    width = x2 - x1
    bridge_height = height + random.randint(5, 12)  # Make bridges more substantial
    
    # Generate organic bridge shape
    organic_mask = generate_organic_shape(width, bridge_height, "bridge")
    
    # Apply the organic shape
    for bridge_y in range(bridge_height):
        for bridge_x in range(width):
            if organic_mask[bridge_y, bridge_x]:
                actual_x = x1 + bridge_x
                actual_y = y + bridge_y
                # Add some texture to the bridge
                bridge_intensity = fill + np.random.normal(0, 6)
                bridge_intensity = max(20, min(120, int(bridge_intensity)))
                try:
                    draw.point((actual_x, actual_y), fill=bridge_intensity)
                except:
                    continue

def generate_image(defect_type, img_size, num_lines, line_width, gap_width):
    """Generate SEM-like image with enhanced roughness"""
    # Start with slightly varied background
    background_base = 180
    background_variation = np.random.normal(background_base, 5, (img_size, img_size))
    background_variation = np.clip(background_variation, 160, 200).astype(np.uint8)
    
    img = Image.fromarray(background_variation, mode="L")
    draw = ImageDraw.Draw(img)
    
    # Calculate starting position
    start_x = (img_size - (num_lines * line_width + (num_lines - 1) * gap_width)) // 2
    
    # More varied intensity for lines
    intensity_base = int(np.clip(np.random.normal(50, 10), 30, 80))
    
    # Select defect parameters
    defect_line_idx = random.randint(1, num_lines - 2)
    defect_y = random.randint(img_size // 4, 3 * img_size // 4)
    defect_height = random.randint(12, 25)  # Slightly larger defects

    # Draw lines with enhanced roughness
    for i in range(num_lines):
        x1 = start_x + i * (line_width + gap_width)
        x2 = x1 + line_width
        
        # Vary intensity per line for more realism
        line_intensity = intensity_base + np.random.normal(0, 8)
        line_intensity = max(20, min(100, int(line_intensity)))
        
        draw_realistic_rough_line(draw, x1, 0, x2, img_size, fill=line_intensity, line_width=line_width)

        # Add defects with organic shapes
        if defect_type == "gap" and i == defect_line_idx:
            draw_organic_gap(draw, x1, x2, defect_y, defect_height, fill=background_base)

        if defect_type == "bridge" and i == defect_line_idx:
            bx1 = x2
            bx2 = bx1 + gap_width
            bridge_intensity = line_intensity + np.random.normal(0, 5)
            draw_organic_bridge(draw, bx1, bx2, defect_y, 
                              height=random.randint(3, 6), 
                              fill=int(max(20, min(100, bridge_intensity))))

    # Convert to numpy for post-processing
    img_np = np.array(img).astype(np.float32)
    
    # Add subtle realistic noise
    noise_strength = 6  # Reduced from 12
    noise = np.random.normal(0, noise_strength, (img_size, img_size))
    img_np = img_np + noise
    
    # Add very subtle surface texture
    img_np = add_surface_texture(img_np, texture_strength=3)  # Reduced from 8
    
    # Apply minimal final noise and clipping
    final_noise = np.random.normal(0, 2, (img_size, img_size))  # Reduced from 3
    img_np = np.clip(img_np + final_noise, 0, 255).astype(np.uint8)
    
    # Convert back to PIL and apply very subtle blur
    img = Image.fromarray(img_np)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.2))  # Reduced from 0.3
    
    return img

def generate_dataset_for_azure_ml(output_dir, csv_file, img_size=128, num_images=200, 
                                  num_lines=8, line_width=6, gap_width=6, azure_run=None):
    """Generate SEM dataset optimized for Azure ML pipeline"""
    
    logger.info("Enhanced SEM Dataset Generator Starting...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Image size: {img_size}x{img_size}")
    logger.info(f"Images per class: {num_images}")
    
    # Log to Azure ML if available
    if azure_run:
        azure_run.log("dataset_img_size", img_size)
        azure_run.log("dataset_num_images_per_class", num_images)
        azure_run.log("dataset_num_lines", num_lines)
    
    classes = ["good", "gap", "bridge"]
    
    # Create directory structure compatible with training script
    data_dir = os.path.join(output_dir, "data")
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        os.makedirs(cls_path, exist_ok=True)
        logger.info(f"Created directory: {cls_path}")
    
    # Create samples directory for quality review
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Generate dataset with progress tracking
    total_images = len(classes) * num_images
    generated_count = 0
    
    # Create the CSV file in the data directory
    csv_path = os.path.join(data_dir, "labels.csv")
    
    with open(csv_path, mode="w", newline="") as label_file:
        writer = csv.writer(label_file)
        writer.writerow(["filename", "label"])

        for label in classes:
            logger.info(f"Generating {label} class images...")
            class_dir = os.path.join(data_dir, label)
            
            for i in range(num_images):
                try:
                    # Generate image
                    defect_type = label if label != "good" else "none"
                    img = generate_image(defect_type, img_size, num_lines, line_width, gap_width)
                    
                    # Save image in class subdirectory
                    fname = f"{label}_{i:04d}.png"
                    fpath = os.path.join(class_dir, fname)
                    img.save(fpath)
                    
                    # Write to CSV with just filename (not full path)
                    writer.writerow([fname, label])
                    
                    generated_count += 1
                    
                    # Progress logging
                    if i % 50 == 0 and i > 0:
                        logger.info(f"Generated {i}/{num_images} {label} images")
                        if azure_run:
                            progress = (generated_count / total_images) * 100
                            azure_run.log("generation_progress", progress)
                
                except Exception as e:
                    logger.error(f"Error generating image {i} for class {label}: {e}")
                    continue

            logger.info(f"Completed {label} class: {num_images} images")

    # Generate sample images for quality review
    logger.info("Generating sample images for review...")
    for label in classes:
        try:
            defect_type = label if label != "good" else "none"
            img = generate_image(defect_type, img_size, num_lines, line_width, gap_width)
            sample_path = os.path.join(samples_dir, f"sample_{label}.png")
            img.save(sample_path)
            logger.info(f"Sample saved: {sample_path}")
        except Exception as e:
            logger.error(f"Error generating sample for {label}: {e}")
    
    # Create dataset metadata
    metadata = {
        "dataset_info": {
            "classes": classes,
            "num_classes": len(classes),
            "images_per_class": num_images,
            "total_images": generated_count,
            "image_size": img_size,
            "generation_parameters": {
                "num_lines": num_lines,
                "line_width": line_width,
                "gap_width": gap_width
            }
        },
        "file_structure": {
            "data_directory": "data/",
            "class_subdirectories": [f"data/{cls}/" for cls in classes],
            "labels_file": "data/labels.csv",
            "samples_directory": "samples/"
        }
    }
    
    metadata_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("✅ Enhanced SEM image generation completed successfully!")
    logger.info(f"📁 Dataset saved in: {data_dir}")
    logger.info(f"📄 Labels CSV saved in: {csv_path}")
    logger.info(f"🔬 Generated {generated_count} total images")
    logger.info(f"🎯 Sample images available in: {samples_dir}")
    logger.info(f"📊 Metadata saved in: {metadata_path}")
    
    if azure_run:
        azure_run.log("total_generated_images", generated_count)
        azure_run.upload_file("dataset_metadata.json", metadata_path)
        
        # Upload sample images
        for label in classes:
            sample_path = os.path.join(samples_dir, f"sample_{label}.png")
            if os.path.exists(sample_path):
                azure_run.upload_file(f"sample_{label}.png", sample_path)
    
    return csv_path, data_dir, metadata_path

def main():
    try:
        parser = argparse.ArgumentParser(description='Enhanced SEM Dataset Generator for Azure ML')
        parser.add_argument('--output_dir', type=str, default='./outputs', 
                            help='Output directory for generated dataset')
        parser.add_argument('--img_size', type=int, default=128, 
                            help='Size of generated images (square)')
        parser.add_argument('--num_images', type=int, default=200, 
                            help='Number of images per class')
        parser.add_argument('--num_lines', type=int, default=8, 
                            help='Number of lines in each image')
        parser.add_argument('--line_width', type=int, default=6, 
                            help='Width of each line')
        parser.add_argument('--gap_width', type=int, default=6, 
                            help='Width of gaps between lines')
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed for reproducibility')
        
        args = parser.parse_args()
        
        # Set random seed for reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)
        
        # Initialize Azure ML run context if available
        azure_run = None
        if AZURE_ML_AVAILABLE:
            try:
                azure_run = Run.get_context()
                logger.info("Azure ML run context initialized")
                
                # Log parameters
                azure_run.log("seed", args.seed)
                azure_run.log("img_size", args.img_size)
                azure_run.log("num_images_per_class", args.num_images)
                
            except Exception as e:
                logger.warning(f"Could not initialize Azure ML context: {e}")
        
        # Generate dataset
        csv_path, data_dir, metadata_path = generate_dataset_for_azure_ml(
            output_dir=args.output_dir,
            csv_file="labels.csv",  # Will be created in data_dir
            img_size=args.img_size,
            num_images=args.num_images,
            num_lines=args.num_lines,
            line_width=args.line_width,
            gap_width=args.gap_width,
            azure_run=azure_run
        )
        
        # Complete Azure ML run if available
        if azure_run:
            azure_run.complete()
        
        return 0

    except Exception as e:
        logger.error("❌ An error occurred during SEM image generation.")
        logger.error(f"Error details: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())