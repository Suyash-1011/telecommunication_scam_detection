#  Voice Phishing Detection System

An AI-powered system for detecting voice phishing (vishing) attacks using machine learning ensemble methods. This project analyzes audio recordings of phone calls to identify potential phishing attempts using advanced audio feature extraction and multiple classification models.

---

##  Team Members

- **Teena Chowdary**
- **Suyash Suryavanshi**
- **Jatin Kushwala**
- **Ayush Pareek**

---


##  Overview

Voice phishing (vishing) is a growing cybersecurity threat where attackers use phone calls to deceive victims into revealing sensitive information. This system provides an automated solution to detect such fraudulent calls by analyzing audio characteristics and patterns that distinguish phishing calls from legitimate ones.

The system employs an ensemble of 2 machine learning models:
- Deep Neural Network (DNN)
- XGBoost


Each model contributes to the final prediction through a voting mechanism, improving overall detection accuracy.

---

##  Features

- **Audio Feature Extraction**: Extracts 35+ audio features including:
  - Mel-Frequency Cepstral Coefficients (MFCCs)
  - Spectral features (centroid, rolloff)
  - Chroma features
  - Mel spectrogram statistics
  - Zero-crossing rate
  - Tempo analysis

- **Data Augmentation**: Expands the dataset using:
  - Time stretching (faster/slower playback)
  - Pitch shifting (higher/lower pitch)
  - White noise injection (simulating poor call quality)
  - Background hum addition (50 Hz electrical hum)
  - Amplitude scaling

- **Ensemble Learning**: Combines predictions from multiple models for robust detection

- **Web Application**: Flask-based REST API with user-friendly interface for real-time predictions

- **Comprehensive Evaluation**: Detailed metrics, ROC curves, and confusion matrices

---

##  Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-repo/voice-phishing-detection.git
cd voice-phishing-detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

```txt
numpy>=1.21.0
pandas>=1.3.0
librosa>=0.9.0
soundfile>=0.10.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
xgboost>=1.5.0
flask>=2.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Step 4: Create Configuration File

Create a `config.py` file in the project root:

```python
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PHISHING_DIR = DATA_DIR / "phishing"
LEGITIMATE_DIR = DATA_DIR / "legitimate"
AUGMENTED_DIR = DATA_DIR / "augmented"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
UPLOAD_FOLDER = BASE_DIR / "uploads"

# Create directories
for dir_path in [PHISHING_DIR, LEGITIMATE_DIR, AUGMENTED_DIR, 
                 PROCESSED_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR, UPLOAD_FOLDER]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Audio processing
SAMPLE_RATE = 22050
N_MFCC = 13
AUGMENT_FACTOR = 5

# Model paths
DNN_MODEL_PATH = MODEL_DIR / "dnn_model.h5"
DNN_SCALER_PATH = MODEL_DIR / "dnn_scaler.pkl"
XGBOOST_MODEL_PATH = MODEL_DIR / "xgboost_model.json"

# DNN hyperparameters
DNN_EPOCHS = 100
DNN_BATCH_SIZE = 32
DNN_VALIDATION_SPLIT = 0.2

# XGBoost hyperparameters
XGBOOST_N_ESTIMATORS = 100
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.1

# Ensemble settings
ENSEMBLE_THRESHOLD = 0.5
DNN_WEIGHT = 0.5
XGBOOST_WEIGHT = 0.5

# Flask settings
FLASK_DEBUG = True
FLASK_PORT = 5001
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload

# Logging
LOG_FILE = LOGS_DIR / "training.log"
```

---


##  Usage

### Step 1: Prepare Dataset

Place your audio files in the appropriate directories:
- Phishing call recordings: `data/phishing/` (as .mp3 files)
- Legitimate call recordings: `data/legitimate/` (as .mp3 files)

**Minimum requirement**: 40 samples per class (80 total) for meaningful results.

### Step 2: Explore Dataset

```bash
python src/01_data_exploration.py
```

This script analyzes your dataset and provides:
- File count per class
- Audio duration statistics
- Sample rate verification

### Step 3: Augment Data

```bash
python src/02_data_augmentation.py
```

Expands your dataset using various augmentation techniques. Default augmentation factor is 5x.

### Step 4: Extract Features

```bash
python src/feature_extraction.py
```

Extracts audio features and creates train/test splits (80/20). This step:
- Processes augmented training data
- Keeps test data as original (no augmentation)
- Saves features to CSV files

### Step 5: Train Models

Train each model sequentially:

```bash
# Train Deep Neural Network
python src/04_train_dnn.py

# Train XGBoost
python src/05_train_xgboost.py


```

### Step 6: Evaluate Ensemble

```bash
python src/06_ensemble_eval.py
```

Generates comprehensive evaluation metrics and visualizations.

### Step 7: Run Web Application

```bash
python src/new_flask_app.py
```

Access the web interface at: `http://localhost:5001`

---

##  Pipeline Workflow

```mermaid
graph TD
    A[Raw Audio Files] --> B[Data Exploration]
    B --> C[Data Augmentation]
    C --> D[Feature Extraction]
    D --> E[Train-Test Split]
    E --> F1[Train DNN]
    E --> F2[Train XGBoost]
    E --> F3[Train Random Forest]
    E --> F4[Train LightGBM]
    F1 --> G[Ensemble Evaluation]
    F2 --> G
    F3 --> G
    F4 --> G
    G --> H[Model Deployment]
    H --> I[Flask Web App]
```

---

##  Models

### 1. Deep Neural Network (DNN)

**Architecture:**
- Input Layer: 35+ features
- Hidden Layer 1: 256 neurons, ReLU, BatchNorm, Dropout (0.4)
- Hidden Layer 2: 128 neurons, ReLU, BatchNorm, Dropout (0.3)
- Hidden Layer 3: 64 neurons, ReLU, BatchNorm, Dropout (0.2)
- Hidden Layer 4: 32 neurons, ReLU, BatchNorm, Dropout (0.2)
- Output Layer: 1 neuron, Sigmoid

**Optimizer:** Adam (lr=0.001)  
**Loss:** Binary Cross-Entropy  
**Callbacks:** Early Stopping, Learning Rate Reduction

### 2. XGBoost

**Configuration:**
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8



### Ensemble Voting

The final prediction is computed as the average probability from all available models:

```
P(phishing) = mean(P_DNN, P_XGBoost, P_RF, P_LightGBM)
```

Classification: PHISHING if P(phishing) > 0.5, else LEGITIMATE

---


## Future Improvements

### 1. Dataset Enhancement
- Collect more diverse phishing call samples (target 500+ per class)
- Include various phishing scenarios (IRS scams, tech support, bank fraud)
- Add calls in multiple languages
- Include calls with different audio qualities

### 2. Feature Engineering
- Add speech-to-text features using ASR models
- Implement Natural Language Processing on transcripts
- Extract prosodic features (speech rate, pauses, stress patterns)
- Include speaker diarization features

### 3. Model Improvements
- Implement Convolutional Neural Networks (CNNs) for spectrogram analysis
- Use Recurrent Neural Networks (LSTMs/GRUs) for temporal patterns
- Try transformer-based audio models (Wav2Vec, HuBERT)
- Implement stacking ensemble instead of simple voting
- Use cross-validated hyperparameter tuning (GridSearchCV)

### 4. Production Readiness
- Add authentication and rate limiting to API
- Implement model versioning and A/B testing
- Add monitoring and alerting for model drift
- Deploy using Docker containers
- Set up CI/CD pipeline
- Add real-time audio streaming support

### 5. User Experience
- Build mobile application for on-device detection
- Add browser extension for VoIP call analysis
- Implement batch processing for call centers
- Create dashboard for call pattern analytics

---
