"""
Generate sample skin disease images for testing the classification system.
Creates realistic synthetic images for different skin conditions.
"""

import os
from PIL import Image, ImageDraw, ImageFilter
import random
import numpy as np

# Create uploads directory if it doesn't exist
UPLOADS_DIR = '/Users/darshu/Projects/skin_disease/skin_disease_project/uploads'
os.makedirs(UPLOADS_DIR, exist_ok=True)

def create_normal_skin(filename):
    """Create a normal, healthy skin image."""
    img = Image.new('RGB', (400, 300), color=(230, 200, 180))
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Add subtle texture
    for _ in range(100):
        x = random.randint(0, 400)
        y = random.randint(0, 300)
        size = random.randint(1, 3)
        color = (random.randint(220, 240), random.randint(190, 210), random.randint(170, 190), 100)
        draw.ellipse([x, y, x+size, y+size], fill=color)
    
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    img.save(os.path.join(UPLOADS_DIR, filename))
    print(f"✓ Created: {filename}")

def create_acne(filename):
    """Create an image with acne-like spots."""
    img = Image.new('RGB', (400, 300), color=(220, 180, 160))
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Add acne spots (small red/dark bumps)
    for _ in range(15):
        x = random.randint(50, 350)
        y = random.randint(50, 250)
        size = random.randint(8, 20)
        # Red center with darker edge
        draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], 
                    fill=(180, 100, 80, 200))
        draw.ellipse([x-size//4, y-size//4, x+size//4, y+size//4], 
                    fill=(200, 120, 100, 180))
    
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img.save(os.path.join(UPLOADS_DIR, filename))
    print(f"✓ Created: {filename}")

def create_eczema(filename):
    """Create an image with eczema-like redness."""
    img = Image.new('RGB', (400, 300), color=(200, 140, 140))
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Large inflamed red areas
    for _ in range(5):
        x = random.randint(30, 350)
        y = random.randint(30, 250)
        w = random.randint(80, 150)
        h = random.randint(60, 120)
        color = (220, 80, 80, 150)
        draw.ellipse([x-w//2, y-h//2, x+w//2, y+h//2], fill=color)
    
    # Add texture
    for _ in range(200):
        x = random.randint(0, 400)
        y = random.randint(0, 300)
        color = (random.randint(190, 230), random.randint(60, 120), 
                random.randint(60, 120), 80)
        draw.point((x, y), fill=color)
    
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    img.save(os.path.join(UPLOADS_DIR, filename))
    print(f"✓ Created: {filename}")

def create_psoriasis(filename):
    """Create an image with psoriasis-like patches."""
    img = Image.new('RGB', (400, 300), color=(210, 160, 150))
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Dark red scaly patches
    for _ in range(8):
        x = random.randint(30, 350)
        y = random.randint(30, 250)
        w = random.randint(70, 140)
        h = random.randint(50, 100)
        # Dark red base
        draw.rectangle([x-w//2, y-h//2, x+w//2, y+h//2], 
                      fill=(160, 60, 60))
        
        # Scaly texture
        for _ in range(30):
            sx = random.randint(x-w//2, x+w//2)
            sy = random.randint(y-h//2, y+h//2)
            draw.ellipse([sx-4, sy-4, sx+4, sy+4], 
                        fill=(180, 80, 80, 180))
    
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    img.save(os.path.join(UPLOADS_DIR, filename))
    print(f"✓ Created: {filename}")

def create_melanoma(filename):
    """Create an image with dark mole/melanoma."""
    img = Image.new('RGB', (400, 300), color=(230, 200, 180))
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Large dark spot
    x, y = random.randint(100, 300), random.randint(80, 220)
    # Outer dark ring
    draw.ellipse([x-60, y-60, x+60, y+60], fill=(40, 20, 20))
    # Medium brown
    draw.ellipse([x-45, y-45, x+45, y+45], fill=(80, 40, 40))
    # Dark center
    draw.ellipse([x-30, y-30, x+30, y+30], fill=(20, 10, 10))
    
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    img.save(os.path.join(UPLOADS_DIR, filename))
    print(f"✓ Created: {filename}")

def create_nevus(filename):
    """Create an image with benign mole."""
    img = Image.new('RGB', (400, 300), color=(230, 200, 180))
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Small brown mole
    for _ in range(3):
        x = random.randint(100, 300)
        y = random.randint(80, 220)
        size = random.randint(30, 50)
        color = (140, 100, 60)
        draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], fill=color)
    
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img.save(os.path.join(UPLOADS_DIR, filename))
    print(f"✓ Created: {filename}")

def create_dermatitis(filename):
    """Create an image with dermatitis."""
    img = Image.new('RGB', (400, 300), color=(215, 175, 165))
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Scattered red areas
    for _ in range(12):
        x = random.randint(40, 360)
        y = random.randint(40, 260)
        w = random.randint(40, 80)
        h = random.randint(30, 70)
        draw.ellipse([x-w//2, y-h//2, x+w//2, y+h//2], 
                    fill=(220, 100, 90, 200))
    
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    img.save(os.path.join(UPLOADS_DIR, filename))
    print(f"✓ Created: {filename}")

# Generate multiple samples for each condition
print("Generating 500+ sample skin disease images...\n")

# Normal skin (80 images)
for i in range(1, 81):
    create_normal_skin(f'01_normal_skin_{i:03d}.jpg')

# Acne (75 images)
for i in range(1, 76):
    create_acne(f'02_acne_{i:03d}.jpg')

# Eczema (80 images)
for i in range(1, 81):
    create_eczema(f'03_eczema_{i:03d}.jpg')

# Psoriasis (75 images)
for i in range(1, 76):
    create_psoriasis(f'04_psoriasis_{i:03d}.jpg')

# Melanoma/Mole (60 images)
for i in range(1, 61):
    create_melanoma(f'05_melanoma_{i:03d}.jpg')

# Benign Nevus (60 images)
for i in range(1, 61):
    create_nevus(f'06_nevus_{i:03d}.jpg')

# Dermatitis (70 images)
for i in range(1, 71):
    create_dermatitis(f'07_dermatitis_{i:03d}.jpg')

print(f"\n✅ Total images created: 500")
print(f"📁 Location: {UPLOADS_DIR}")
print("\nImage breakdown:")
print("  • Normal skin: 80 images")
print("  • Acne: 75 images")
print("  • Eczema: 80 images")
print("  • Psoriasis: 75 images")
print("  • Melanoma: 60 images")
print("  • Benign Nevus: 60 images")
print("  • Dermatitis: 70 images")
print("\nYou can now test the app with these sample images!")
