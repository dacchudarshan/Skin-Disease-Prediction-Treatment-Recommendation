"""
Mistral Vision API Integration for Skin Disease Detection
Provides accurate image identification and analysis using Mistral's vision capabilities
"""

import os
import base64
import json
from typing import Dict, Tuple, List, Any
from pathlib import Path
import requests
from PIL import Image
import io

class MistralVisionAnalyzer:
    """Analyzes skin disease images using Mistral API vision capabilities."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Mistral Vision Analyzer
        
        Args:
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        self.model = "pixtral-12b-2409"
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string for API"""
        with open(image_path, 'rb') as img_file:
            return base64.standard_b64encode(img_file.read()).decode('utf-8')
    
    def analyze_skin_condition(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze skin condition from image using Mistral Vision API
        
        Args:
            image_path: Path to skin image
            
        Returns:
            Dictionary with disease identification and confidence
        """
        try:
            # Encode image
            image_data = self.encode_image_to_base64(image_path)
            
            # Get file extension for media type
            ext = Path(image_path).suffix.lower()
            media_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.webp': 'image/webp'
            }
            media_type = media_type_map.get(ext, 'image/jpeg')
            
            # Prepare prompt for skin disease detection
            prompt = """You are an expert dermatologist analyzing a skin condition image.

IMPORTANT: Analyze this skin image and provide a detailed medical analysis in JSON format:
{
  "disease": "Specific disease/condition name",
  "confidence": 0-100,
  "severity": "Mild/Moderate/Severe",
  "observations": ["observation 1", "observation 2", ...],
  "recommendations": ["recommendation 1", "recommendation 2", ...],
  "accuracy_metrics": {
    "clarity_score": 0-100,
    "visibility_score": 0-100,
    "diagnostic_confidence": 0-100
  }
}

Be specific and medical-accurate."""
            
            # Call Mistral API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.3
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            analysis_text = result['choices'][0]['message']['content']
            
            # Parse the response
            analysis = self._parse_mistral_response(analysis_text)
            
            return {
                'success': True,
                'method': 'Mistral Vision API',
                'analysis': analysis,
                'accuracy': analysis.get('confidence', 0),
                'model': self.model,
                'raw_response': analysis_text
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"API request failed: {str(e)}",
                'method': 'Mistral Vision API'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Analysis failed: {str(e)}",
                'method': 'Mistral Vision API'
            }
    
    def _parse_mistral_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Mistral API response and extract structured data"""
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                return {
                    'disease': 'Unable to parse',
                    'confidence': 0,
                    'observations': [response_text[:200]],
                    'severity': 'Unknown',
                    'recommendations': ['Consult a dermatologist'],
                    'raw_text': response_text
                }
        except json.JSONDecodeError:
            return {
                'disease': 'Analysis Incomplete',
                'confidence': 0,
                'observations': [response_text[:200]],
                'severity': 'Unknown',
                'recommendations': ['Consult a dermatologist'],
                'raw_text': response_text
            }


class BatchMistralAnalyzer:
    """Handles batch processing of multiple images with Mistral Vision API"""
    
    def __init__(self, api_key: str = None):
        """Initialize batch analyzer"""
        self.analyzer = MistralVisionAnalyzer(api_key)
    
    def analyze_batch(self, image_paths: List[str], show_progress: bool = True) -> Dict[str, Any]:
        """
        Analyze multiple images in batch
        
        Args:
            image_paths: List of image file paths
            show_progress: Whether to show progress
            
        Returns:
            Batch analysis results
        """
        results = {
            'total_images': len(image_paths),
            'successful': 0,
            'failed': 0,
            'analyses': [],
            'average_accuracy': 0,
            'timestamp': str(Path(image_paths[0]).stat().st_mtime) if image_paths else None
        }
        
        accuracies = []
        
        for idx, image_path in enumerate(image_paths):
            if show_progress:
                print(f"Processing image {idx + 1}/{len(image_paths)}: {image_path}")
            
            try:
                analysis = self.analyzer.analyze_skin_condition(image_path)
                results['analyses'].append({
                    'image': image_path,
                    'analysis': analysis
                })
                
                if analysis['success']:
                    results['successful'] += 1
                    acc = analysis.get('accuracy', 0)
                    if acc > 0:
                        accuracies.append(acc)
                else:
                    results['failed'] += 1
                    
            except Exception as e:
                results['failed'] += 1
                results['analyses'].append({
                    'image': image_path,
                    'error': str(e)
                })
        
        if accuracies:
            results['average_accuracy'] = sum(accuracies) / len(accuracies)
        
        return results
