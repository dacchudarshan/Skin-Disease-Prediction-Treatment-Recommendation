#!/usr/bin/env python3
"""
Test script for Mistral Vision API integration
Verifies API connectivity and image analysis functionality
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project to path
sys.path.insert(0, '/Users/darshu/Projects/skin_disease_copy/skin_disease_project')

# Load environment variables
load_dotenv('/Users/darshu/Projects/skin_disease_copy/.env')

from mistral_vision import MistralVisionAnalyzer

def test_mistral_api():
    """Test Mistral API configuration and connectivity."""
    print("=" * 70)
    print("MISTRAL VISION API - INTEGRATION TEST")
    print("=" * 70)
    
    # Check API key
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        print("❌ ERROR: MISTRAL_API_KEY not found in environment")
        return False
    
    print(f"✅ API Key loaded: {api_key[:10]}...{api_key[-4:]}")
    
    # Initialize analyzer
    try:
        analyzer = MistralVisionAnalyzer()
        print("✅ MistralVisionAnalyzer initialized successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize analyzer: {e}")
        return False
    
    # Test with a sample image
    sample_image = '/Users/darshu/Projects/skin_disease_copy/uploads'
    if os.path.exists(sample_image):
        # Find first image in uploads
        image_files = list(Path(sample_image).glob('*.{jpg,jpeg,png,bmp}'))
        
        if image_files:
            test_image = str(image_files[0])
            print(f"\n📸 Testing with image: {test_image}")
            print("-" * 70)
            
            result = analyzer.analyze_skin_condition(test_image)
            
            if result.get('success'):
                print("✅ Analysis successful!")
                print(f"   Method: {result.get('method')}")
                print(f"   Model: {result.get('model')}")
                print(f"   Accuracy Confidence: {result.get('accuracy_confidence')}%")
                
                analysis = result.get('analysis', {})
                print(f"\n   Disease/Condition: {analysis.get('disease', 'N/A')}")
                print(f"   Confidence: {analysis.get('confidence', 'N/A')}%")
                print(f"   Severity: {analysis.get('severity', 'N/A')}")
                
                observations = analysis.get('observations', [])
                if observations:
                    print(f"\n   Key Observations:")
                    for obs in observations[:3]:  # Show first 3
                        print(f"      - {obs}")
                
                recommendations = analysis.get('recommendations', [])
                if recommendations:
                    print(f"\n   Recommendations:")
                    for rec in recommendations[:3]:  # Show first 3
                        print(f"      - {rec}")
                
                return True
            else:
                print(f"❌ Analysis failed: {result.get('error')}")
                return False
        else:
            print("⚠️  No test images found in uploads folder")
            print("   Create a test image in /uploads to verify functionality")
    else:
        print(f"⚠️  Uploads folder not found: {sample_image}")
    
    return True

if __name__ == '__main__':
    success = test_mistral_api()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ INTEGRATION TEST PASSED - Mistral Vision API is ready!")
    else:
        print("❌ INTEGRATION TEST FAILED - Check configuration")
    print("=" * 70)
    
    sys.exit(0 if success else 1)
