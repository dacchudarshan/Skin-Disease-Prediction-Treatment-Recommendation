"""
Localization & Accessibility System
Provides multi-language support, text-to-speech, accessibility features,
cultural adaptation, and low-bandwidth optimization
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en-US"
    SPANISH = "es-ES"
    FRENCH = "fr-FR"
    GERMAN = "de-DE"
    PORTUGUESE = "pt-BR"
    MANDARIN = "zh-CN"
    HINDI = "hi-IN"
    ARABIC = "ar-SA"
    RUSSIAN = "ru-RU"
    JAPANESE = "ja-JP"


class AccessibilityFeature(Enum):
    """Accessibility features."""
    TEXT_TO_SPEECH = "Text-to-Speech"
    HIGH_CONTRAST = "High Contrast Mode"
    LARGE_FONT = "Large Font Size"
    SCREEN_READER = "Screen Reader Compatible"
    VOICE_CONTROL = "Voice Control"
    KEYBOARD_NAVIGATION = "Keyboard Navigation"
    CAPTION_SUPPORT = "Captions/Subtitles"
    COLOR_BLIND_MODE = "Color Blind Mode"


class TextDirection(Enum):
    """Text direction for languages."""
    LTR = "left-to-right"
    RTL = "right-to-left"


@dataclass
class LocalizationString:
    """Localized text string."""
    key: str
    language: Language
    text: str
    context: Optional[str] = None
    pluralization_rules: Optional[Dict] = None
    
    def to_dict(self):
        return {
            'key': self.key,
            'language': self.language.value,
            'text': self.text,
            'context': self.context
        }


class LocalizationEngine:
    """Multi-language localization system."""
    
    def __init__(self):
        """Initialize localization engine."""
        self.supported_languages = {lang.name: lang.value for lang in Language}
        self.translation_database = self._build_translation_database()
        self.current_language = Language.ENGLISH
    
    def _build_translation_database(self) -> Dict:
        """Build comprehensive translation database."""
        return {
            'en-US': {
                'app_title': 'Skin Disease Detection & Treatment',
                'nav_home': 'Home',
                'nav_analysis': 'Image Analysis',
                'nav_history': 'Medical History',
                'nav_appointments': 'Appointments',
                'nav_treatment': 'Treatment Plans',
                'btn_upload': 'Upload Image',
                'btn_analyze': 'Analyze',
                'btn_save': 'Save',
                'btn_cancel': 'Cancel',
                'result_diagnosis': 'Diagnosis',
                'result_confidence': 'Confidence Level',
                'result_recommendation': 'Recommendation',
                'error_invalid_image': 'Invalid image format',
                'error_upload_failed': 'Upload failed',
                'success_analysis_complete': 'Analysis complete',
                'warning_needs_specialist': 'Specialist consultation recommended',
                'label_symptoms': 'Current Symptoms',
                'label_duration': 'Duration of Condition',
                'label_treatments': 'Current Treatments',
                'hint_upload_clear_image': 'Upload a clear, well-lit image for best results',
                'info_dermatologist_contact': 'Contact a dermatologist for professional evaluation'
            },
            'es-ES': {
                'app_title': 'Detección y Tratamiento de Enfermedades de la Piel',
                'nav_home': 'Inicio',
                'nav_analysis': 'Análisis de Imagen',
                'nav_history': 'Historial Médico',
                'nav_appointments': 'Citas',
                'nav_treatment': 'Planes de Tratamiento',
                'btn_upload': 'Subir Imagen',
                'btn_analyze': 'Analizar',
                'btn_save': 'Guardar',
                'btn_cancel': 'Cancelar',
                'result_diagnosis': 'Diagnóstico',
                'result_confidence': 'Nivel de Confianza',
                'result_recommendation': 'Recomendación',
                'error_invalid_image': 'Formato de imagen inválido',
                'error_upload_failed': 'Error en la carga',
                'success_analysis_complete': 'Análisis completado',
                'warning_needs_specialist': 'Se recomienda consulta especializada',
                'label_symptoms': 'Síntomas Actuales',
                'label_duration': 'Duración de la Condición',
                'label_treatments': 'Tratamientos Actuales',
                'hint_upload_clear_image': 'Sube una imagen clara y bien iluminada para mejores resultados',
                'info_dermatologist_contact': 'Contacta a un dermatólogo para evaluación profesional'
            },
            'pt-BR': {
                'app_title': 'Detecção e Tratamento de Doenças de Pele',
                'nav_home': 'Início',
                'nav_analysis': 'Análise de Imagem',
                'nav_history': 'Histórico Médico',
                'nav_appointments': 'Consultas',
                'nav_treatment': 'Planos de Tratamento',
                'btn_upload': 'Fazer Upload',
                'btn_analyze': 'Analisar',
                'btn_save': 'Salvar',
                'btn_cancel': 'Cancelar',
                'result_diagnosis': 'Diagnóstico',
                'result_confidence': 'Nível de Confiança',
                'result_recommendation': 'Recomendação',
                'error_invalid_image': 'Formato de imagem inválido',
                'error_upload_failed': 'Falha no upload',
                'success_analysis_complete': 'Análise concluída',
                'warning_needs_specialist': 'Consulta especializada recomendada',
                'label_symptoms': 'Sintomas Atuais',
                'label_duration': 'Duração da Condição',
                'label_treatments': 'Tratamentos Atuais',
                'hint_upload_clear_image': 'Faça upload de uma imagem clara e bem iluminada para melhores resultados',
                'info_dermatologist_contact': 'Entre em contato com um dermatologista para avaliação profissional'
            },
            'fr-FR': {
                'app_title': 'Détection et Traitement des Maladies de la Peau',
                'nav_home': 'Accueil',
                'nav_analysis': 'Analyse d\'Image',
                'nav_history': 'Antécédents Médicaux',
                'nav_appointments': 'Rendez-vous',
                'nav_treatment': 'Plans de Traitement',
                'btn_upload': 'Télécharger',
                'btn_analyze': 'Analyser',
                'btn_save': 'Enregistrer',
                'btn_cancel': 'Annuler',
                'result_diagnosis': 'Diagnostic',
                'result_confidence': 'Niveau de Confiance',
                'result_recommendation': 'Recommandation',
                'error_invalid_image': 'Format d\'image invalide',
                'error_upload_failed': 'Erreur de téléchargement',
                'success_analysis_complete': 'Analyse terminée',
                'warning_needs_specialist': 'Consultation spécialisée recommandée',
                'label_symptoms': 'Symptômes Actuels',
                'label_duration': 'Durée de la Condition',
                'label_treatments': 'Traitements Actuels',
                'hint_upload_clear_image': 'Téléchargez une image claire et bien éclairée pour de meilleurs résultats',
                'info_dermatologist_contact': 'Contactez un dermatologue pour une évaluation professionnelle'
            },
            'hi-IN': {
                'app_title': 'त्वचा रोग पहचान और उपचार',
                'nav_home': 'होम',
                'nav_analysis': 'छवि विश्लेषण',
                'nav_history': 'चिकित्सा इतिहास',
                'nav_appointments': 'नियुक्तियां',
                'nav_treatment': 'उपचार योजनाएं',
                'btn_upload': 'अपलोड करें',
                'btn_analyze': 'विश्लेषण करें',
                'btn_save': 'सहेजें',
                'btn_cancel': 'रद्द करें',
                'result_diagnosis': 'निदान',
                'result_confidence': 'आत्मविश्वास स्तर',
                'result_recommendation': 'सिफारिश',
                'error_invalid_image': 'अमान्य छवि प्रारूप',
                'error_upload_failed': 'अपलोड विफल',
                'success_analysis_complete': 'विश्लेषण पूर्ण',
                'warning_needs_specialist': 'विशेषज्ञ परामर्श की सिफारिश की जाती है',
                'label_symptoms': 'वर्तमान लक्षण',
                'label_duration': 'स्थिति की अवधि',
                'label_treatments': 'वर्तमान उपचार',
                'hint_upload_clear_image': 'सर्वोत्तम परिणामों के लिए एक स्पष्ट, अच्छी रोशनी वाली छवि अपलोड करें',
                'info_dermatologist_contact': 'पेशेवर मूल्यांकन के लिए एक त्वचा विशेषज्ञ से संपर्क करें'
            },
            'ar-SA': {
                'app_title': 'كشف وعلاج أمراض الجلد',
                'nav_home': 'الرئيسية',
                'nav_analysis': 'تحليل الصور',
                'nav_history': 'السجل الطبي',
                'nav_appointments': 'المواعيد',
                'nav_treatment': 'خطط العلاج',
                'btn_upload': 'تحميل',
                'btn_analyze': 'تحليل',
                'btn_save': 'حفظ',
                'btn_cancel': 'إلغاء',
                'result_diagnosis': 'التشخيص',
                'result_confidence': 'مستوى الثقة',
                'result_recommendation': 'التوصية',
                'error_invalid_image': 'صيغة صورة غير صحيحة',
                'error_upload_failed': 'فشل التحميل',
                'success_analysis_complete': 'اكتمل التحليل',
                'warning_needs_specialist': 'يوصى باستشارة متخصص',
                'label_symptoms': 'الأعراض الحالية',
                'label_duration': 'مدة الحالة',
                'label_treatments': 'العلاجات الحالية',
                'hint_upload_clear_image': 'قم بتحميل صورة واضحة وجيدة الإضاءة للحصول على أفضل النتائج',
                'info_dermatologist_contact': 'اتصل بطبيب جلدية لإجراء تقييم احترافي'
            }
        }
    
    def get_string(self, key: str, language: Optional[Language] = None) -> str:
        """Get localized string."""
        lang = language or self.current_language
        lang_code = lang.value
        
        if lang_code in self.translation_database:
            return self.translation_database[lang_code].get(key, key)
        
        # Fallback to English
        return self.translation_database['en-US'].get(key, key)
    
    def get_all_strings(self, language: Optional[Language] = None) -> Dict:
        """Get all strings for language."""
        lang = language or self.current_language
        return self.translation_database.get(lang.value, {})
    
    def set_language(self, language: Language) -> bool:
        """Set current language."""
        if language in Language:
            self.current_language = language
            logger.info(f"Language changed to {language.value}")
            return True
        return False


class AccessibilityManager:
    """Manages accessibility features."""
    
    def __init__(self):
        """Initialize accessibility manager."""
        self.enabled_features = []
        self.text_to_speech_enabled = False
        self.high_contrast_enabled = False
        self.large_font_enabled = False
        self.screen_reader_enabled = False
        self.voice_control_enabled = False
    
    def enable_feature(self, feature: AccessibilityFeature) -> bool:
        """Enable accessibility feature."""
        if feature not in self.enabled_features:
            self.enabled_features.append(feature)
            logger.info(f"Accessibility feature enabled: {feature.value}")
            return True
        return False
    
    def disable_feature(self, feature: AccessibilityFeature) -> bool:
        """Disable accessibility feature."""
        if feature in self.enabled_features:
            self.enabled_features.remove(feature)
            logger.info(f"Accessibility feature disabled: {feature.value}")
            return True
        return False
    
    def is_feature_enabled(self, feature: AccessibilityFeature) -> bool:
        """Check if feature is enabled."""
        return feature in self.enabled_features
    
    def get_accessibility_settings(self) -> Dict:
        """Get current accessibility settings."""
        return {
            'enabled_features': [f.value for f in self.enabled_features],
            'text_to_speech': AccessibilityFeature.TEXT_TO_SPEECH in self.enabled_features,
            'high_contrast': AccessibilityFeature.HIGH_CONTRAST in self.enabled_features,
            'large_font': AccessibilityFeature.LARGE_FONT in self.enabled_features,
            'screen_reader': AccessibilityFeature.SCREEN_READER in self.enabled_features,
            'voice_control': AccessibilityFeature.VOICE_CONTROL in self.enabled_features
        }


class TextToSpeechManager:
    """Manages text-to-speech functionality."""
    
    def __init__(self):
        """Initialize TTS manager."""
        self.available_voices = {
            'en-US': ['male', 'female'],
            'es-ES': ['male', 'female'],
            'fr-FR': ['male', 'female'],
            'pt-BR': ['male', 'female'],
            'hi-IN': ['male', 'female'],
            'ar-SA': ['male', 'female']
        }
        self.current_voice = 'female'
        self.speech_rate = 1.0
    
    def text_to_speech(self, text: str, language: Language = Language.ENGLISH,
                      voice: str = 'female', rate: float = 1.0) -> Dict:
        """Convert text to speech."""
        
        lang_code = language.value
        
        if lang_code not in self.available_voices:
            return {'success': False, 'error': 'Language not supported for TTS'}
        
        return {
            'success': True,
            'text': text,
            'language': lang_code,
            'voice': voice,
            'speech_rate': rate,
            'audio_file': f'audio_{hash(text)}.mp3',
            'duration_seconds': len(text) / 150  # Rough estimate
        }
    
    def set_voice(self, voice: str) -> bool:
        """Set TTS voice."""
        self.current_voice = voice
        return True
    
    def set_speech_rate(self, rate: float) -> bool:
        """Set speech rate (0.5 = half speed, 1.0 = normal, 2.0 = double)."""
        if 0.5 <= rate <= 2.0:
            self.speech_rate = rate
            return True
        return False


class LowBandwidthOptimization:
    """Optimizes content for low-bandwidth environments."""
    
    @staticmethod
    def compress_image(image_path: str, quality: int = 70) -> str:
        """Compress image for low bandwidth."""
        return f"compressed_{image_path}"
    
    @staticmethod
    def create_text_only_mode(data: Dict) -> Dict:
        """Create text-only version of content."""
        return {
            'mode': 'text-only',
            'disable_images': True,
            'disable_videos': True,
            'reduce_color_palette': True,
            'compress_css': True,
            'data_size_kb': 50
        }
    
    @staticmethod
    def estimate_bandwidth_requirement(content_type: str) -> Dict:
        """Estimate bandwidth requirements."""
        requirements = {
            'image_analysis': {'mb': 2, 'time_seconds': 5},
            'video_consultation': {'mb': 50, 'time_seconds': 300},
            'text_report': {'mb': 0.5, 'time_seconds': 1},
            'form_submission': {'mb': 1, 'time_seconds': 2}
        }
        return requirements.get(content_type, {'mb': 5, 'time_seconds': 10})


class CulturalAdaptation:
    """Adapts content for cultural context."""
    
    def __init__(self):
        """Initialize cultural adaptation."""
        self.cultural_contexts = {
            'en-US': {'date_format': 'MM/DD/YYYY', 'time_format': '12h', 'currency': 'USD'},
            'es-ES': {'date_format': 'DD/MM/YYYY', 'time_format': '24h', 'currency': 'EUR'},
            'pt-BR': {'date_format': 'DD/MM/YYYY', 'time_format': '24h', 'currency': 'BRL'},
            'fr-FR': {'date_format': 'DD/MM/YYYY', 'time_format': '24h', 'currency': 'EUR'},
            'hi-IN': {'date_format': 'DD/MM/YYYY', 'time_format': '24h', 'currency': 'INR'},
            'ar-SA': {'date_format': 'DD/MM/YYYY', 'time_format': '24h', 'currency': 'SAR'}
        }
    
    def get_date_format(self, language: Language) -> str:
        """Get date format for language."""
        lang_code = language.value
        return self.cultural_contexts.get(lang_code, {}).get('date_format', 'MM/DD/YYYY')
    
    def get_time_format(self, language: Language) -> str:
        """Get time format for language."""
        lang_code = language.value
        return self.cultural_contexts.get(lang_code, {}).get('time_format', '12h')
    
    def get_currency(self, language: Language) -> str:
        """Get currency for language."""
        lang_code = language.value
        return self.cultural_contexts.get(lang_code, {}).get('currency', 'USD')


# Create singleton instances
localization_engine = LocalizationEngine()
accessibility_manager = AccessibilityManager()
text_to_speech = TextToSpeechManager()
low_bandwidth = LowBandwidthOptimization()
cultural_adaptation = CulturalAdaptation()
