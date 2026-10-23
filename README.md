# 🌾 Sindh Smart Farmer AI — Backend

AI-powered agricultural advisory backend for small-scale farmers in Sindh, Pakistan. Provides real-time multilingual crop disease diagnosis and farming guidance in **Urdu**, **Sindhi**, and **English**.

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Node.js Express Gateway (Port 3000)    │
│  ┌─────────────┐  ┌────────────┐  ┌───────────┐ │
│  │ OpenRouter   │  │  Gemini    │  │ Local SAU │ │
│  │ Cloud LLM   │→ │  2.0 Flash │→ │ Knowledge │ │
│  │ (Llama 70B) │  │  (Backup)  │  │  Engine   │ │
│  └─────────────┘  └────────────┘  └───────────┘ │
│              ↕                                    │
│  ┌─────────────────────────────────────────────┐ │
│  │  PyTorch SindhAgriNet AI (Port 5001)        │ │
│  │  775,706 params • 100% Acc • 25 Epochs      │ │
│  │  BiLSTM + Self-Attention + Multi-Head CLS   │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Gateway | Node.js + Express |
| Cloud LLM | OpenRouter API (Llama 3.3 70B Instruct) |
| Deep Learning | PyTorch 2.4.0 (CPU) — SindhAgriNet |
| ML API | FastAPI + Uvicorn (Port 5001) |
| Languages | Urdu (اردو), Sindhi (سنڌي), English |

## Trained PyTorch Model

- **Name:** SindhAgriNet v1.0
- **Architecture:** Bidirectional LSTM + Self-Attention + Multi-Task Classification Heads
- **Parameters:** 775,706
- **Disease Classes:** Cotton Leaf Curl Virus, Rice Blast, Sugarcane Red Rot, Wheat Yellow Rust, Chilli Leaf Curl, Tomato Blight, Mango Malformation, Healthy Crop
- **Intent Classes:** Diagnosis, Treatment, Prevention, Fertilizer, Irrigation, General Advice
- **Training Accuracy:** 100%
- **Final Loss:** 0.0004

## API Endpoints

### Node.js (Port 3000)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze-crop` | Multimodal crop disease analysis |
| POST | `/api/chat` | Multilingual farming chat (Urdu/Sindhi/English) |
| GET | `/api/demo-cases` | Sample demo cases |
| GET | `/api/health` | Health check |
| GET | `/api/config` | Active engine configuration |
| GET | `/api/pytorch-model-status` | PyTorch model metrics |

### Python FastAPI (Port 5001)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/model/predict` | PyTorch crop disease prediction |
| POST | `/api/model/chat` | PyTorch multilingual chat inference |
| GET | `/api/model/metrics` | Training metrics & model info |
| GET | `/health` | Service health check |

## Setup

```bash
# 1. Install Node.js dependencies
npm install

# 2. Create .env file
cp .env.example .env
# Add your OPENROUTER_API_KEY

# 3. Install Python dependencies
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn pillow

# 4. Train PyTorch model (optional — pretrained weights included)
cd ai-backend
python train.py

# 5. Start PyTorch AI service
cd ai-backend
python -m uvicorn app:app --host 127.0.0.1 --port 5001

# 6. Start Node.js server
node server.js
```

## Files

```
├── server.js              # Node.js Express backend (main gateway)
├── package.json           # Node.js dependencies
├── .env.example           # Environment variables template
├── ai-backend/
│   ├── model.py           # SindhAgriNet PyTorch architecture
│   ├── dataset.py         # Training data & tokenizer
│   ├── train.py           # PyTorch training script
│   ├── inference.py       # Inference & multilingual reasoning engine
│   ├── app.py             # FastAPI REST API server
│   ├── vocab.json         # Multilingual vocabulary (478 tokens)
│   ├── metadata.json      # Training metrics
│   └── sindh_cropnet_model.pth  # Trained model weights (3.1 MB)
```

## Author

Built for Sindh's farming community 🇵🇰
