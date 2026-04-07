# Skin Disease Analysis System

A comprehensive web-based application for analyzing and classifying skin diseases using image processing and machine learning techniques.

## Features

- **Image Upload & Analysis**: Upload skin disease images for automated analysis
- **Disease Classification**: Classify skin conditions using advanced deep learning models
- **Image Processing**: Comprehensive image validation and preprocessing
- **Batch Analysis**: Process multiple images at once
- **Reports & Analytics**: Generate detailed reports on analysis results
- **Data Visualization**: Visual comparison and statistical analysis of results
- **PDF Export**: Export analysis results as PDF documents
- **Gallery Management**: Browse and manage uploaded images
- **User Management**: Secure user authentication and data management

## System Architecture

```
skin_disease_project/
├── app.py                          # Main Flask application
├── app_mistral.py                  # Mistral AI integration
├── deep_learning_models.py         # ML model definitions
├── advanced_accuracy.py            # Advanced detection algorithms
├── user_management.py              # User authentication & management
├── telemedicine_api.py             # Telemedicine integration
├── treatment_recommendations.py    # Treatment suggestions
├── analytics_reporting.py          # Data analytics & reporting
├── mistral_vision.py               # Mistral Vision API integration
├── optimize.py                     # Performance optimization
├── templates/                      # HTML templates
│   ├── index.html                  # Home page
│   ├── batch_analysis.html         # Batch processing
│   ├── comparison.html             # Image comparison
│   ├── disease_info.html           # Disease information
│   ├── gallery.html                # Image gallery
│   ├── mistral_analysis.html       # AI-powered analysis
│   ├── statistics.html             # Statistics dashboard
├── static/                         # Static files
│   └── css/                        # Stylesheets
│       ├── home.css
│       ├── theme.css
│       └── components.css
├── uploads/                        # User uploaded images
├── logs/                           # Application logs
└── models/                         # Pre-trained ML models
```

## Requirements

- Python 3.8+
- Flask 2.3+
- TensorFlow/Keras (for ML models)
- OpenCV (for image processing)
- Pillow (for image handling)
- NumPy & SciPy (for numerical operations)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd original_skin_disease_final
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Running the Application

### Development Mode
```bash
source venv/bin/activate
export FLASK_APP=skin_disease_project/app.py
export FLASK_ENV=development
python -m flask run
```

The application will be available at `http://127.0.0.1:5000`

### Production Mode
```bash
gunicorn --config skin_disease_project/gunicorn_config.py skin_disease_project.app:app
```

## Usage

1. **Upload Image**: Click on the upload section to select a skin disease image
2. **Analyze**: The system will process the image and provide analysis
3. **View Results**: Review the classification, confidence scores, and recommendations
4. **Export**: Download results as PDF or compare with other images

## API Endpoints

### Core Analysis
- `POST /api/analyze` - Analyze a single image
- `POST /api/batch-analyze` - Analyze multiple images
- `GET /api/results/<id>` - Retrieve analysis results

### Image Management
- `GET /api/gallery` - List uploaded images
- `DELETE /api/image/<id>` - Remove an image

### User Management
- `POST /api/auth/login` - User login
- `POST /api/auth/register` - User registration
- `GET /api/user/profile` - Get user profile

## Configuration

Configure the application via environment variables in `.env`:

```env
FLASK_ENV=development
DEBUG=True
SECRET_KEY=your-secret-key
DATABASE_URL=postgresql://user:password@localhost/dbname
UPLOAD_FOLDER=uploads/
MAX_UPLOAD_SIZE=16777216
MODEL_PATH=models/
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Test specific modules:
```bash
python test_skin_validation.py
python test_mistral_integration.py
```

## Features in Development

- [ ] Real-time video analysis
- [ ] Mobile app support
- [ ] Advanced AI-powered recommendations
- [ ] Integration with medical databases
- [ ] Multi-language support
- [ ] Telemedicine consultation booking

## Performance Optimization

The application includes several optimization modules:
- `optimize.py` - Model and inference optimization
- `advanced_accuracy.py` - Enhanced accuracy detection
- Caching mechanisms for faster response times
- Batch processing for large datasets

## Security

- User authentication with bcrypt password hashing
- JWT token-based authorization
- CSRF protection
- Input validation and sanitization
- Secure file upload handling
- Database encryption for sensitive data

## Support

For issues, feature requests, or contributions, please:
1. Check existing issues on GitHub
2. Create a detailed issue report
3. Fork and submit pull requests

## Citation

If you use this system in your research, please cite:
```
@software{skin_disease_analysis_2024,
  title={Skin Disease Analysis System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/skin-disease-analysis}
}
```

## Disclaimer

This system is for research and educational purposes only. It should not be used as a replacement for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice.

## Acknowledgments

- TensorFlow/Keras community
- OpenCV community
- Medical imaging datasets
- Contributors and testers

## Changelog

### Version 1.0.0 (2024)
- Initial release
- Core image analysis functionality
- User management system
- Report generation
- Batch processing capabilities

## Future Roadmap

- Q3 2024: Mobile application
- Q4 2024: Advanced AI integration
- Q1 2025: Telemedicine features
- Q2 2025: International expansion
