"""
LLM Advisor Service
Generates medical recommendations using LLM APIs (Gemini or OpenAI).
"""
import logging
from abc import ABC, abstractmethod

from app.config import get_settings, DISEASE_INFO

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMAdvisorBase(ABC):
    """Abstract base class for LLM advisors."""
    
    @abstractmethod
    async def generate_recommendations(
        self, 
        disease: str, 
        confidence: float
    ) -> dict[str, str]:
        """Generate recommendations for a detected disease."""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if LLM is loaded and ready."""
        pass


def get_system_prompt() -> str:
    """Get the system prompt for the LLM."""
    return """You are a helpful medical AI assistant specializing in dermatology. 
Your role is to provide informational guidance about skin conditions based on AI image analysis.

IMPORTANT GUIDELINES:
1. Always emphasize that this is AI-generated information, NOT a medical diagnosis
2. Recommend consulting a dermatologist for proper evaluation
3. Be empathetic and reassuring while being factually accurate
4. Provide practical, actionable advice
5. For serious conditions (melanoma, BCC), emphasize urgency of professional consultation
6. Keep responses concise but comprehensive

You will receive information about a detected skin condition and must provide:
1. Recommendations: General treatment approaches and care options
2. Next Steps: What the person should do next
3. Tips: Daily care and prevention tips"""


def get_user_prompt(disease: str, confidence: float, disease_info: dict) -> str:
    """Generate the user prompt for the LLM."""
    severity = disease_info.get("severity", "unknown")
    contagious = "contagious" if disease_info.get("contagious", False) else "not contagious"
    description = disease_info.get("description", "")
    
    confidence_level = "high" if confidence > 0.85 else "moderate" if confidence > 0.6 else "low"
    
    return f"""Based on AI image analysis, the following skin condition was detected:

**Detected Condition:** {disease}
**Confidence Level:** {confidence_level} ({confidence:.1%})
**Severity Category:** {severity}
**Contagious:** {contagious}
**Description:** {description}

Please provide:
1. **RECOMMENDATIONS**: What treatment options and care approaches are typically recommended for this condition? (2-3 sentences)
2. **NEXT_STEPS**: What should the person do next? Consider the severity and confidence level. (2-3 sentences)
3. **TIPS**: What daily care tips and prevention measures would help? (2-3 bullet points)

Format your response EXACTLY as:
RECOMMENDATIONS: [your recommendations here]
NEXT_STEPS: [your next steps here]
TIPS: [your tips here]"""


def parse_llm_response(response: str) -> dict[str, str]:
    """Parse LLM response into structured format."""
    result = {
        "recommendations": "",
        "next_steps": "",
        "tips": ""
    }
    
    # Try to parse structured response
    lines = response.strip().split('\n')
    current_key = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        upper_line = line.upper()
        if upper_line.startswith('RECOMMENDATIONS:'):
            if current_key:
                result[current_key] = ' '.join(current_content).strip()
            current_key = 'recommendations'
            current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
        elif upper_line.startswith('NEXT_STEPS:') or upper_line.startswith('NEXT STEPS:'):
            if current_key:
                result[current_key] = ' '.join(current_content).strip()
            current_key = 'next_steps'
            current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
        elif upper_line.startswith('TIPS:'):
            if current_key:
                result[current_key] = ' '.join(current_content).strip()
            current_key = 'tips'
            current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
        elif current_key:
            current_content.append(line)
    
    # Save last section
    if current_key:
        result[current_key] = ' '.join(current_content).strip()
    
    # If parsing failed, use the whole response
    if not any(result.values()):
        result["recommendations"] = response[:500] if len(response) > 500 else response
        result["next_steps"] = "Please consult a dermatologist for proper evaluation and treatment plan."
        result["tips"] = "Keep the affected area clean and avoid scratching. Monitor for changes."
    
    return result


def _get_fallback_response(disease: str, confidence: float, disease_info: dict) -> dict[str, str]:
    """Generate fallback response when LLM is unavailable."""
    severity = disease_info.get("severity", "unknown")
    
    if severity == "serious":
        urgency = "It is strongly recommended to consult a dermatologist as soon as possible."
    elif severity in ["mild-moderate", "moderate"]:
        urgency = "Consider scheduling an appointment with a dermatologist for proper evaluation."
    else:
        urgency = "While typically benign, monitoring the condition and consulting a healthcare provider if changes occur is advisable."
    
    return {
        "recommendations": f"Based on the analysis suggesting {disease}, {disease_info.get('description', '')} Treatment options vary depending on individual cases and should be determined by a healthcare professional.",
        "next_steps": f"{urgency} Bring any relevant medical history and note when you first noticed the condition.",
        "tips": "• Keep the affected area clean and moisturized\n• Avoid scratching or irritating the skin\n• Protect from excessive sun exposure\n• Document any changes with photos for medical consultations"
    }


class GeminiLLM(LLMAdvisorBase):
    """Google Gemini API-based LLM advisor."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        self._loaded = False
    
    def load_model(self):
        if self._loaded:
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            # Use gemini-2.0-flash or gemini-1.5-flash-latest
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self._loaded = True
            logger.info("Gemini client initialized with gemini-2.0-flash")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
    
    async def generate_recommendations(
        self, 
        disease: str, 
        confidence: float
    ) -> dict[str, str]:
        if not self._loaded:
            self.load_model()
        
        disease_info = DISEASE_INFO.get(disease, {})
        
        try:
            prompt = f"{get_system_prompt()}\n\n{get_user_prompt(disease, confidence, disease_info)}"
            response = await self.model.generate_content_async(prompt)
            return parse_llm_response(response.text)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return _get_fallback_response(disease, confidence, disease_info)
    
    def is_loaded(self) -> bool:
        return self._loaded


class OpenAILLM(LLMAdvisorBase):
    """OpenAI API-based LLM advisor."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self._loaded = False
    
    def load_model(self):
        if self._loaded:
            return
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
            self._loaded = True
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
    
    async def generate_recommendations(
        self, 
        disease: str, 
        confidence: float
    ) -> dict[str, str]:
        if not self._loaded:
            self.load_model()
        
        disease_info = DISEASE_INFO.get(disease, {})
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": get_user_prompt(disease, confidence, disease_info)}
                ],
                max_tokens=512,
                temperature=0.7
            )
            return parse_llm_response(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return _get_fallback_response(disease, confidence, disease_info)
    
    def is_loaded(self) -> bool:
        return self._loaded


# Singleton instance
_llm_advisor: LLMAdvisorBase | None = None


def get_llm_advisor() -> LLMAdvisorBase:
    """Get or create singleton LLM advisor instance based on configuration."""
    global _llm_advisor
    
    if _llm_advisor is None:
        provider = settings.llm_provider.lower()
        
        if provider == "openai" and settings.openai_api_key:
            _llm_advisor = OpenAILLM(settings.openai_api_key)
        elif provider == "gemini" and settings.google_api_key:
            _llm_advisor = GeminiLLM(settings.google_api_key)
        else:
            # Default to Gemini if no valid provider
            logger.warning(f"LLM provider '{provider}' not configured properly. Check API keys.")
            if settings.google_api_key:
                _llm_advisor = GeminiLLM(settings.google_api_key)
            elif settings.openai_api_key:
                _llm_advisor = OpenAILLM(settings.openai_api_key)
            else:
                raise ValueError("No LLM API key configured. Set GOOGLE_API_KEY or OPENAI_API_KEY.")
        
        logger.info(f"LLM advisor created: {type(_llm_advisor).__name__}")
    
    return _llm_advisor


def preload_llm():
    """Preload the LLM at startup."""
    advisor = get_llm_advisor()
    if hasattr(advisor, 'load_model'):
        advisor.load_model()
