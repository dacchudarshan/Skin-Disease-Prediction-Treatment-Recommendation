#!/usr/bin/env python
"""
Test script to demonstrate skin photo validation feature.
"""

import sys
import os
from pathlib import Path

# Add the skin_disease_project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'skin_disease_project'))

from app import is_skin_photo

def test_validation(image_path):
    """Test if image passes skin photo validation."""
    print(f"\nTesting: {image_path}")
    print("-" * 60)
    
    if not os.path.exists(image_path):
        print(f"✗ File does not exist: {image_path}")
        return
    
    is_skin, message = is_skin_photo(image_path)
    
    if is_skin:
        print(f"✓ Valid skin photo: {message}")
    else:
        print(f"✗ Invalid photo: {message}")
    print("-" * 60)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SKIN PHOTO VALIDATION TEST")
    print("="*60)
    
    # Look for sample images in uploads folder
    uploads_dir = os.path.join(os.path.dirname(__file__), 'skin_disease_project', 'uploads')
    
    if os.path.exists(uploads_dir):
        print(f"\nFound uploads directory: {uploads_dir}")
        image_files = [f for f in os.listdir(uploads_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        if image_files:
            print(f"Found {len(image_files)} image(s) to test:\n")
            for img_file in image_files[:5]:  # Test first 5 images
                test_validation(os.path.join(uploads_dir, img_file))
        else:
            print("No image files found in uploads folder.")
            print("\nTo test with actual images:")
            print("1. Run the Flask app: python skin_disease_project/app.py")
            print("2. Upload images via the web interface")
            print("3. Images will be validated and accepted/rejected based on:")
            print("   - Skin tone color ranges")
            print("   - Image clarity (Laplacian variance)")
            print("   - Color saturation levels")
            print("   - Color variation (not a solid color image)")
    else:
        print(f"Uploads directory not found: {uploads_dir}")
        print("\nValidation Features:")
        print("-" * 60)
        print("✓ Skin Tone Detection: Checks for R > G > B with appropriate ranges")
        print("✓ Blur Detection: Ensures image is clear (Laplacian variance > 50)")
        print("✓ Saturation Check: Rejects over-saturated images (likely not real photos)")
        print("✓ Color Variety: Rejects images with too few colors (< 100 unique colors)")
        print("✓ Size Check: Rejects images smaller than 100x100 pixels")
        print("\nRejection reasons include:")
        print("  • Image too small")
        print("  • Image too blurry")
        print("  • Unnatural color saturation")
        print("  • Insufficient color variation")
        print("  • Non-skin tone colors")
        print("-" * 60)
        
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print("The app now validates that uploaded images are skin photos before analysis.")
    print("This prevents non-skin images (animals, objects, drawings, etc.) from being")
    print("analyzed as skin disease photos.")
    print("="*60 + "\n")
