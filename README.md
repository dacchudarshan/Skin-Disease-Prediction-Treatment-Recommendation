# Skin Disease Prediction & Treatment Recommendation

A comprehensive web application for predicting skin diseases using deep learning models and providing evidence-based treatment recommendations powered by AI.

## Features

- **Disease Detection**: Advanced deep learning models for accurate skin disease identification
- **AI-Powered Recommendations**: Treatment recommendations using Mistral AI integration
- **Multiple Analysis Methods**: 
  - Single image analysis
  - Batch processing for multiple images
  - Comparison analysis
- **Telemedicine Support**: API endpoints for telemedicine integration
- **Analytics & Reporting**: Detailed statistics and performance metrics
- **User Management**: Secure user authentication and management
- **Accessibility**: Localization and accessibility features
- **Security Compliance**: Built-in security and compliance measures

## Requirements

- Python 3.8+
- TensorFlow/PyTorch (for deep learning models)
- Flask (web framework)
- Mistral AI API access (optional, for AI recommendations)
- Required packages in `requirements.txt`

## Installation

### 1. Clone or Download the Repository
```bash
cd "skin disease prediction & treatment recommendation"
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory with:
```
MISTRAL_API_KEY=your_api_key_here
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
```

## Usage

### Running the Web Application

**Standard Application:**
```bash
python skin-disease/app.py
```

**With Mistral AI Integration:**
```bash
python skin-disease/app_mistral.py
```

**Optimized Version:**
```bash
python skin-disease/app_optimized.py
```

**Using Gunicorn (Production):**
```bash
gunicorn -c skin-disease/gunicorn_config.py skin-disease:app
```

The application will be available at `http://localhost:5000`

### Starting Mistral Service
```bash
bash start_mistral.sh
```

### Running Tests
```bash
# Test Mistral integration
python test_mistral_integration.py

# Test skin validation
python test_skin_validation.py
```

### Data Cleaning
To remove non-skin disease images from your dataset:
```bash
python cleanup_non_skin_images.py
```

### Generate Sample Data
```bash
python generate_samples.py
```

## API Endpoints

### Web Interface
- `GET /` - Home page
- `POST /analyze` - Single image analysis
- `POST /batch-analyze` - Batch image analysis
- `GET /disease/<disease_id>` - Disease information
- `GET /gallery` - Image gallery
- `GET /statistics` - Analytics dashboard

### Telemedicine API
- `POST /api/telemedicine/diagnosis` - Submit diagnosis request
- `GET /api/telemedicine/results/<request_id>` - Get diagnosis results
- See `telemedicine_api.py` for detailed documentation

## Model Information

The application uses advanced deep learning models for skin disease detection:
- **Accuracy Optimization**: See `advanced_accuracy.py` for model improvement techniques
- **Model Training**: Training scripts available for custom model development
- **Supported Diseases**: Multiple skin disease categories with treatment recommendations

## Key Components

### Treatment Recommendations (`treatment_recommendations.py`)
- Evidence-based treatment suggestions
- Integration with medical databases
- Personalized recommendations based on analysis

### Mistral AI Vision (`mistral_vision.py`)
- Advanced image analysis capabilities
- Natural language generation for reports
- Multi-language support

### Analytics & Reporting (`analytics_reporting.py`)
- Disease statistics and trends
- Performance metrics
- Usage analytics

### Security (`security_compliance.py`)
- HIPAA compliance measures
- Data encryption
- Secure API authentication

## Development

### Running in Development Mode
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python skin-disease/app.py
```

### Quick Start Integration
For rapid testing and integration:
```bash
python skin-disease/quick_start_integration.py
```

## Performance Optimization

For production deployments, use the optimized version:
```bash
python skin-disease/app_optimized.py
```

Features include:
- Model caching
- Request batching
- Output compression
- Improved database queries

## Troubleshooting

### Model Loading Issues
- Ensure model files are in `skin-disease/models/`
- Check TensorFlow/PyTorch installation
- Verify file permissions

### Mistral API Errors
- Verify API key is set in environment variables
- Check network connectivity
- Review API rate limits

### Image Upload Problems
- Ensure `skin-disease/uploads/` directory has write permissions
- Check file size limits in Flask configuration
- Verify supported image formats (JPEG, PNG)

## Contributing

Contributions are welcome! Please:
1. Create a new branch for your feature
2. Write tests for new functionality
3. Update documentation
4. Submit a pull request

## License

This project is licensed under the terms specified in the LICENSE file.

## Contact & Support

For questions or issues:
- Check the troubleshooting section
- Review test files for usage examples
- Check logs in `skin-disease/logs/` directory

## Technologies Used

- **Backend**: Flask, Python
- **ML/DL**: TensorFlow/PyTorch
- **AI**: Mistral AI API
- **Frontend**: HTML, CSS, JavaScript
- **Database**: SQLite/PostgreSQL (configurable)
- **Deployment**: Gunicorn, Docker-ready

## Disclaimer

This application is for educational and informational purposes. It should not be used as a substitute for professional medical advice. Always consult with qualified healthcare professionals for diagnosis and treatment decisions.

---

**Version**: 1.0  
**Last Updated**: january 2026
