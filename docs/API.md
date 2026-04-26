# API Documentation

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: Configure based on your deployment

## Authentication

Currently, the API does not require authentication. For production deployments, consider adding API key authentication.

---

## Endpoints

### POST /analyze_skin

Analyze a skin image for disease detection and get AI-generated recommendations.

**Request**

- **Content-Type**: `multipart/form-data`
- **Body**: 
  - `file` (required): Image file (JPEG, PNG, WebP, BMP)
  - Maximum file size: 10MB

**Example Request**

```bash
curl -X POST "http://localhost:8000/analyze_skin" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@skin_image.jpg"
```

**Response (200 OK)**

```json
{
  "disease": "Eczema",
  "confidence": 0.9234,
  "recommendations": "Based on the analysis suggesting Eczema, this chronic inflammatory skin condition is characterized by itchy, red, dry, and cracked skin. Treatment options typically include moisturizers, topical corticosteroids, and antihistamines for itching. The choice of treatment depends on severity and should be determined by a healthcare professional.",
  "next_steps": "Consider scheduling an appointment with a dermatologist for proper evaluation. Bring any relevant medical history and note when you first noticed the condition. A professional can confirm the diagnosis and recommend appropriate treatment.",
  "tips": "• Keep the affected area clean and moisturized with fragrance-free products\n• Avoid scratching - keep nails short and consider wearing cotton gloves at night\n• Identify and avoid triggers such as certain soaps, detergents, or stress\n• Use lukewarm water for bathing and pat skin dry gently",
  "severity": "mild-moderate",
  "disclaimer": "This is an AI-generated analysis for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Please consult a qualified dermatologist for proper evaluation."
}
```

**Error Responses**

| Status Code | Description |
|-------------|-------------|
| 400 | Invalid image file or format |
| 500 | Internal server error |

```json
{
  "detail": "Invalid file type. Allowed: .jpg, .jpeg, .png, .webp, .bmp"
}
```

---

### GET /health

Check the health status of the API and its components.

**Example Request**

```bash
curl http://localhost:8000/health
```

**Response (200 OK)**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "llm_loaded": true,
  "database_connected": true
}
```

---

### GET /diseases

Get information about all supported skin disease classes.

**Example Request**

```bash
curl http://localhost:8000/diseases
```

**Response (200 OK)**

```json
{
  "diseases": [
    {
      "name": "Eczema",
      "severity": "mild-moderate",
      "contagious": false,
      "description": "A chronic inflammatory skin condition causing itchy, red, dry, and cracked skin."
    },
    {
      "name": "Melanoma",
      "severity": "serious",
      "contagious": false,
      "description": "A serious form of skin cancer that develops from melanocytes. Early detection is critical."
    },
    // ... more diseases
  ],
  "total": 10
}
```

---

### GET /history

Retrieve recent skin analysis history from the database.

**Query Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 10 | Number of results to return (max 100) |
| `offset` | int | 0 | Number of results to skip |

**Example Request**

```bash
curl "http://localhost:8000/history?limit=5&offset=0"
```

**Response (200 OK)**

```json
[
  {
    "id": 42,
    "disease": "Eczema",
    "confidence": 0.9234,
    "recommendations": "...",
    "next_steps": "...",
    "tips": "...",
    "created_at": "2026-04-21T10:30:00Z",
    "image_hash": "a1b2c3d4e5f6..."
  }
]
```

---

### GET /

Root endpoint with API information.

**Example Request**

```bash
curl http://localhost:8000/
```

**Response (200 OK)**

```json
{
  "name": "Skin Disease Detection API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

## API Versioning

The API is also available under the `/api/v1` prefix:

- `POST /api/v1/analyze_skin`
- `GET /api/v1/health`
- `GET /api/v1/diseases`
- `GET /api/v1/history`

---

## Response Schema Reference

### AnalysisResponse

| Field | Type | Description |
|-------|------|-------------|
| `disease` | string | Detected skin disease name |
| `confidence` | float | Confidence score (0-1) |
| `recommendations` | string | Treatment recommendations |
| `next_steps` | string | Recommended actions |
| `tips` | string | Daily care tips |
| `severity` | string | Severity level (benign, mild, mild-moderate, serious) |
| `disclaimer` | string | Medical disclaimer |

### HealthResponse

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | API status ("healthy") |
| `version` | string | API version |
| `model_loaded` | boolean | Classification model status |
| `llm_loaded` | boolean | LLM model status |
| `database_connected` | boolean | Database connection status |

### DiseaseInfo

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Disease name |
| `severity` | string | Severity level |
| `contagious` | boolean | Whether condition is contagious |
| `description` | string | Brief description |

---

## Error Handling

All errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

| Code | Meaning |
|------|---------|
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource doesn't exist |
| 500 | Internal Server Error |

---

## Rate Limiting

Currently no rate limiting is implemented. For production, consider adding rate limiting using libraries like `slowapi`.

---

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide interactive API exploration and testing capabilities.
