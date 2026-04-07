#!/usr/bin/env python
"""
Delete non-skin images from uploads folder.
Keeps only valid skin-related photos.
"""

import os
import sys
from pathlib import Path

# Add the skin_disease_project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'skin_disease_project'))

from app import is_skin_photo

def cleanup_non_skin_images():
    """Delete non-skin images from uploads folder."""
    uploads_dir = os.path.join(os.path.dirname(__file__), 'skin_disease_project', 'uploads')
    
    if not os.path.exists(uploads_dir):
        print(f"Uploads directory not found: {uploads_dir}")
        return
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    image_files = [f for f in os.listdir(uploads_dir) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print("No images found in uploads folder.")
        return
    
    print("\n" + "="*70)
    print("CLEANING NON-SKIN IMAGES FROM UPLOADS")
    print("="*70)
    print(f"Found {len(image_files)} image(s) to process\n")
    
    deleted_count = 0
    kept_count = 0
    
    for img_file in image_files:
        filepath = os.path.join(uploads_dir, img_file)
        is_skin, message = is_skin_photo(filepath)
        
        if is_skin:
            print(f"✓ KEEP: {img_file}")
            print(f"  → {message}\n")
            kept_count += 1
        else:
            print(f"✗ DELETE: {img_file}")
            print(f"  → {message}")
            try:
                os.remove(filepath)
                print(f"  → Deleted successfully\n")
                deleted_count += 1
            except Exception as e:
                print(f"  → Error deleting: {str(e)}\n")
    
    print("="*70)
    print("CLEANUP SUMMARY")
    print("="*70)
    print(f"Total images processed: {len(image_files)}")
    print(f"Skin images kept: {kept_count}")
    print(f"Non-skin images deleted: {deleted_count}")
    print("="*70 + "\n")

if __name__ == "__main__":
    cleanup_non_skin_images()
