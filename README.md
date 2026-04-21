# Skin Disease Detection & LLM Advisor System

A real-time AI-powered system that analyzes skin images, detects diseases using deep learning, and provides personalized recommendations using Large Language Models.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)

## Features

- **Image Analysis**: Upload skin images for AI-powered disease classification
- **10 Disease Classes**: Supports detection of common skin conditions
- **LLM Recommendations**: Get AI-generated treatment recommendations, next steps, and tips
- **Real-time API**: Fast response times with async processing
- **Modern UI**: Clean Streamlit interface with confidence visualization
- **History Tracking**: Store and retrieve past analyses (PostgreSQL)
- **Docker Ready**: Full containerized deployment with Docker Compose

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│    FastAPI      │────▶│   PostgreSQL    │
│   Frontend      │     │    Backend      │     │   Database      │
│   (Port 8501)   │     │   (Port 8000)   │     │   (Port 5432)   │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
              ┌─────▼─────┐ ┌────▼────┐ ┌─────▼─────┐
              │  Image    │ │Efficient│ │   LLM     │
              │Processor  │ │Net-B0   │ │ Advisor   │
              │           │ │Classifier│ │(HF/OpenAI)│
              └───────────┘ └─────────┘ └───────────┘
```

## Supported Skin Conditions

| Disease | Severity | Contagious |
|---------|----------|------------|
| Eczema | Mild-Moderate | No |
| Melanoma | Serious | No |
| Atopic Dermatitis | Mild-Moderate | No |
| Basal Cell Carcinoma (BCC) | Serious | No |
| Melanocytic Nevi (NV) | Benign | No |
| Benign Keratosis-like Lesions (BKL) | Benign | No |
| Psoriasis and Lichen Planus | Mild-Moderate | No |
| Seborrheic Keratoses and Benign Tumors | Benign | No |
| Tinea Ringworm and Fungal Infections | Mild | Yes |
| Warts Molluscum and Viral Infections | Mild | Yes |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- (Optional) NVIDIA GPU with CUDA for faster inference
- (Optional) Kaggle account for dataset download

### 1. Clone the Repository

```bash
git clone <repository-url>
cd skin-disease-detection
```

### 2. Set Up Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# - Set LLM_PROVIDER (huggingface, openai, or gemini)
# - Add API keys if using OpenAI/Gemini
```

### 3. Run with Docker Compose

```bash
# Production mode
docker-compose up --build

# Development mode (with hot-reload)
docker-compose -f docker-compose.dev.yml up --build
```

### 4. Access the Application

- **Frontend UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Development Setup

### Local Development (without Docker)

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend (new terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### Training the Model

1. **Download the Dataset**

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (requires Kaggle API key)
kaggle datasets download -d ismailpromus/skin-diseases-image-dataset -p data/
unzip data/skin-diseases-image-dataset.zip -d data/
```

2. **Train the Model**

```bash
cd backend
python -m ml.train --data-dir ../data --output-dir ml/model_weights

# Options:
# --epochs-phase1 10   # Epochs for frozen base training
# --epochs-phase2 20   # Epochs for fine-tuning
```

3. **Evaluate the Model**

```bash
python -m ml.evaluate --model ml/model_weights/efficientnet_skin_disease.keras --data-dir ../data
```

## API Usage

### Analyze Skin Image

```bash
curl -X POST "http://localhost:8000/analyze_skin" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/skin_image.jpg"
```

**Response:**
```json
{
  "disease": "Eczema",
  "confidence": 0.92,
  "recommendations": "Based on the analysis suggesting Eczema...",
  "next_steps": "Consider scheduling an appointment with a dermatologist...",
  "tips": "• Keep the affected area clean and moisturized...",
  "severity": "mild-moderate",
  "disclaimer": "This is an AI-generated analysis..."
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

### List Supported Diseases

```bash
curl http://localhost:8000/diseases
```

See [API Documentation](docs/API.md) for complete endpoint reference.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://postgres:postgres@db:5432/skin_disease_db` |
| `LLM_PROVIDER` | LLM provider (huggingface, openai, gemini) | `huggingface` |
| `HF_MODEL_NAME` | HuggingFace model name | `mistralai/Mistral-7B-Instruct-v0.3` |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI) | - |
| `GOOGLE_API_KEY` | Google API key (if using Gemini) | - |
| `MODEL_PATH` | Path to trained model | `ml/model_weights/efficientnet_skin_disease.keras` |
| `DEBUG` | Enable debug mode | `false` |

### LLM Provider Options

1. **HuggingFace (Local)** - Default, requires GPU for best performance
   - `Mistral-7B-Instruct-v0.3` - Recommended for GPU
   - `Llama-3.2-3B-Instruct` - Lighter alternative

2. **OpenAI API** - Fast, requires API key
   - Uses `gpt-4o-mini` for cost efficiency

3. **Google Gemini** - Fast, requires API key
   - Uses `gemini-1.5-flash`

## Project Structure

```
skin-disease-detection/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Configuration
│   │   ├── models/              # Pydantic & SQLAlchemy models
│   │   ├── routers/             # API endpoints
│   │   └── services/            # Business logic
│   │       ├── image_processor.py
│   │       ├── classifier.py
│   │       └── llm_advisor.py
│   ├── ml/
│   │   ├── train.py             # Model training
│   │   ├── evaluate.py          # Model evaluation
│   │   └── model_weights/       # Saved models
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py                   # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── data/                        # Dataset (not in git)
├── docs/
│   └── API.md                   # API documentation
├── docker-compose.yml           # Production deployment
├── docker-compose.dev.yml       # Development deployment
├── .env.example                 # Environment template
└── README.md
```

## Technical Details

### Model Architecture

- **Base Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Input Size**: 224×224×3
- **Custom Head**: GlobalAveragePooling → BatchNorm → Dropout(0.3) → Dense(256, ReLU) → BatchNorm → Dropout(0.2) → Dense(10, Softmax)
- **Training**: Two-phase transfer learning
  - Phase 1: Frozen base, train head (10 epochs, lr=1e-3)
  - Phase 2: Fine-tune top 20 layers (20 epochs, lr=1e-5)

### Performance Targets

- Classification Accuracy: >85%
- API Response Time: <5 seconds (including LLM)
- Supported Image Formats: JPEG, PNG, WebP, BMP

## Disclaimer

⚠️ **Medical Disclaimer**: This tool is for **informational and educational purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for proper evaluation of any skin condition. AI predictions may not be accurate and should not be relied upon for medical decisions.

## License

This project is for educational purposes. Please ensure compliance with the dataset license when using for any purpose.

## Acknowledgments

- Dataset: [Skin Diseases Image Dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset) on Kaggle
- Model: EfficientNet-B0 from TensorFlow/Keras
- LLM: HuggingFace Transformers
