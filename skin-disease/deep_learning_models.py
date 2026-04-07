"""
Advanced Deep Learning Module for Skin Disease Detection
Implements CNN, ResNet50, MobileNet, and EfficientNet architectures
with transfer learning, GPU acceleration, and feature extraction capabilities
"""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers, models, preprocessing
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0, EfficientNetB1
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from pathlib import Path
import json
import pickle
from datetime import datetime
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DeepLearningModel(ABC):
    """Abstract base class for all deep learning models."""
    
    def __init__(self, model_name: str, input_shape: tuple = (224, 224, 3), 
                 num_classes: int = 7, use_gpu: bool = True):
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.model = None
        self.history = None
        self.feature_extractor = None
        self.class_names = []
        
        # Configure GPU if available
        if use_gpu:
            self._setup_gpu()
    
    def _setup_gpu(self):
        """Setup GPU acceleration if available."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU initialized: {len(gpus)} GPU(s) found")
            except RuntimeError as e:
                logger.warning(f"GPU setup warning: {e}")
        else:
            logger.info("No GPU available, running on CPU")
    
    @abstractmethod
    def build_model(self):
        """Build the neural network architecture."""
        pass
    
    @abstractmethod
    def get_feature_extractor(self):
        """Create a feature extraction layer."""
        pass
    
    def compile_model(self, optimizer='adam', learning_rate=0.001):
        """Compile the model with specified optimizer."""
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(),
                    keras.metrics.Recall(),
                    keras.metrics.AUC()]
        )
        logger.info(f"Model compiled: {self.model_name}")
    
    def train(self, train_data, val_data, epochs=100, batch_size=32, 
              early_stopping_patience=15):
        """Train the model with data augmentation and callbacks."""
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'models/{self.model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2]
        )
        
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"Training completed for {self.model_name}")
        return self.history
    
    def extract_features(self, image_data: np.ndarray) -> np.ndarray:
        """Extract deep features from input image(s)."""
        if self.feature_extractor is None:
            self.feature_extractor = self.get_feature_extractor()
        
        if len(image_data.shape) == 3:
            image_data = np.expand_dims(image_data, axis=0)
        
        features = self.feature_extractor.predict(image_data, verbose=0)
        return features
    
    def predict(self, image_data: np.ndarray, return_features=False):
        """Make prediction on image."""
        if len(image_data.shape) == 3:
            image_data = np.expand_dims(image_data, axis=0)
        
        # Normalize image
        image_data = image_data.astype('float32') / 255.0
        
        predictions = self.model.predict(image_data, verbose=0)[0]
        
        result = {
            'predictions': {self.class_names[i]: float(predictions[i]) 
                          for i in range(len(self.class_names))},
            'top_3': sorted(zip(self.class_names, predictions), 
                           key=lambda x: x[1], reverse=True)[:3]
        }
        
        if return_features:
            result['features'] = self.extract_features(image_data).tolist()
        
        return result
    
    def save_model(self, path: str):
        """Save model and metadata."""
        self.model.save(f'{path}/{self.model_name}.h5')
        
        metadata = {
            'model_name': self.model_name,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'created_at': datetime.now().isoformat()
        }
        
        with open(f'{path}/{self.model_name}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Model saved to {path}/{self.model_name}")
    
    def load_model(self, path: str):
        """Load model and metadata."""
        self.model = keras.models.load_model(f'{path}/{self.model_name}.h5')
        
        with open(f'{path}/{self.model_name}_metadata.json', 'r') as f:
            metadata = json.load(f)
            self.class_names = metadata['class_names']
        
        logger.info(f"Model loaded from {path}/{self.model_name}")


class CustomCNN(DeepLearningModel):
    """Custom Convolutional Neural Network from scratch."""
    
    def build_model(self):
        """Build custom CNN architecture optimized for skin disease detection."""
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        logger.info("Custom CNN model built")
        return model
    
    def get_feature_extractor(self):
        """Return feature extractor (up to global average pooling)."""
        feature_extractor = models.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-4].output
        )
        return feature_extractor


class ResNet50Transfer(DeepLearningModel):
    """ResNet50 with transfer learning."""
    
    def build_model(self):
        """Build ResNet50 transfer learning model."""
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            input_shape=self.input_shape,
            include_top=False
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        self.base_model = base_model
        logger.info("ResNet50 transfer learning model built")
        return model
    
    def fine_tune(self, num_layers_to_unfreeze=50):
        """Unfreeze and fine-tune base model layers."""
        self.base_model.trainable = True
        
        for layer in self.base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
        
        logger.info(f"Fine-tuning enabled for last {num_layers_to_unfreeze} layers")
    
    def get_feature_extractor(self):
        """Return feature extractor."""
        feature_extractor = models.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-3].output
        )
        return feature_extractor


class MobileNetTransfer(DeepLearningModel):
    """MobileNet with transfer learning for mobile deployment."""
    
    def build_model(self):
        """Build MobileNetV2 transfer learning model."""
        base_model = MobileNetV2(
            weights='imagenet',
            input_shape=self.input_shape,
            include_top=False
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        self.base_model = base_model
        logger.info("MobileNetV2 transfer learning model built")
        return model
    
    def get_feature_extractor(self):
        """Return lightweight feature extractor."""
        feature_extractor = models.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-2].output
        )
        return feature_extractor


class EfficientNetTransfer(DeepLearningModel):
    """EfficientNet with transfer learning - state-of-the-art accuracy/efficiency."""
    
    def build_model(self):
        """Build EfficientNetB0/B1 transfer learning model."""
        base_model = EfficientNetB1(
            weights='imagenet',
            input_shape=self.input_shape,
            include_top=False
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        self.base_model = base_model
        logger.info("EfficientNetB1 transfer learning model built")
        return model
    
    def get_feature_extractor(self):
        """Return feature extractor."""
        feature_extractor = models.Model(
            inputs=self.model.input,
            outputs=self.model.layers[-3].output
        )
        return feature_extractor


class EnsembleDeepLearning:
    """Ensemble multiple deep learning models for improved accuracy."""
    
    def __init__(self, model_list: list, weights: list = None):
        """Initialize ensemble with list of models."""
        self.models = model_list
        self.num_models = len(model_list)
        
        if weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            self.weights = weights
    
    def predict(self, image_data: np.ndarray) -> dict:
        """Make ensemble prediction."""
        if len(image_data.shape) == 3:
            image_data = np.expand_dims(image_data, axis=0)
        
        image_data = image_data.astype('float32') / 255.0
        
        ensemble_predictions = {}
        individual_predictions = []
        
        # Get predictions from all models
        for i, model in enumerate(self.models):
            pred = model.predict(image_data)
            individual_predictions.append(pred['predictions'])
            
            for disease, prob in pred['predictions'].items():
                if disease not in ensemble_predictions:
                    ensemble_predictions[disease] = 0
                ensemble_predictions[disease] += prob * self.weights[i]
        
        # Normalize ensemble predictions
        total = sum(ensemble_predictions.values())
        ensemble_predictions = {
            disease: prob / total for disease, prob in ensemble_predictions.items()
        }
        
        return {
            'ensemble_predictions': ensemble_predictions,
            'individual_predictions': individual_predictions,
            'top_3': sorted(ensemble_predictions.items(), 
                          key=lambda x: x[1], reverse=True)[:3],
            'models': [m.model_name for m in self.models]
        }


class NeuralNetworkFineTuning:
    """Advanced fine-tuning strategies for transfer learning models."""
    
    def __init__(self, model: DeepLearningModel):
        self.model = model
        self.fine_tune_history = []
    
    def gradual_unfreezing(self, train_data, val_data, 
                          unfreeze_stages=[10, 25, 50, 100]):
        """Gradually unfreeze layers during training."""
        base_model = self.model.base_model if hasattr(self.model, 'base_model') else None
        
        if base_model is None:
            logger.warning("Model doesn't support gradual unfreezing")
            return
        
        for stage, num_unfrozen in enumerate(unfreeze_stages):
            logger.info(f"Stage {stage + 1}: Unfreezing {num_unfrozen} layers")
            
            # Unfreeze layers
            for layer in base_model.layers[-num_unfrozen:]:
                layer.trainable = True
            
            # Use lower learning rate for fine-tuning
            lr = 0.0001 / (2 ** stage)
            self.model.compile_model(learning_rate=lr)
            
            # Train for fewer epochs
            history = self.model.train(
                train_data, 
                val_data, 
                epochs=20,
                early_stopping_patience=5
            )
            
            self.fine_tune_history.append({
                'stage': stage,
                'history': history,
                'lr': lr
            })
    
    def learning_rate_finder(self, train_data, start_lr=1e-7, end_lr=1e-1):
        """Find optimal learning rate using exponential schedule."""
        initial_weights = self.model.model.get_weights()
        
        num_batches = len(train_data)
        lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_batches)
        losses = []
        
        for batch, (x, y) in enumerate(train_data):
            lr = lrs[batch]
            self.model.model.optimizer.learning_rate.assign(lr)
            
            loss = self.model.model.train_on_batch(x, y)
            losses.append(loss)
        
        # Reset weights
        self.model.model.set_weights(initial_weights)
        
        # Find learning rate with steepest slope
        best_idx = np.argmin(np.gradient(losses))
        best_lr = lrs[best_idx]
        
        logger.info(f"Recommended learning rate: {best_lr:.6f}")
        return best_lr, lrs, losses


# Disease classification class names
DEFAULT_DISEASE_CLASSES = [
    'Acne/Rosacea',
    'Eczema/Dermatitis',
    'Psoriasis',
    'Fungal Infection',
    'Viral Infection',
    'Melanoma/Mole (High Risk)',
    'Other/Normal'
]


def create_model(model_type: str = 'efficientnet', **kwargs) -> DeepLearningModel:
    """Factory function to create deep learning models."""
    models_dict = {
        'cnn': CustomCNN,
        'resnet50': ResNet50Transfer,
        'mobilenet': MobileNetTransfer,
        'efficientnet': EfficientNetTransfer
    }
    
    ModelClass = models_dict.get(model_type.lower(), EfficientNetTransfer)
    model = ModelClass(**kwargs)
    model.build_model()
    model.class_names = kwargs.get('class_names', DEFAULT_DISEASE_CLASSES)
    
    return model
