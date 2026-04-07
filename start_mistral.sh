#!/usr/bin/env bash
# Quick Start Guide for Mistral Vision API Integration

echo "🚀 Starting Skin Disease Detection with Mistral Vision API"
echo "============================================================"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please create .env file with: MISTRAL_API_KEY=your_key"
    exit 1
fi

echo "✅ Environment configuration found"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

echo "✅ Dependencies installed"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $PYTHON_VERSION"

# Test Mistral integration
echo ""
echo "🧪 Testing Mistral integration..."
python test_mistral_integration.py

# Start the server
echo ""
echo "🌐 Starting Flask server..."
echo "============================================================"
echo "Access the application at: http://localhost:5000"
echo "Try Mistral Analysis at: http://localhost:5000/mistral"
echo "============================================================"
echo ""

cd skin_disease_project
python app.py
