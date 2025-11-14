# app/flask_app_new.py - MINIMAL WORKING VERSION

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tensorflow as tf
import pickle
from xgboost import XGBClassifier
from flask import Flask, render_template, request, jsonify
import os
import traceback

# Import your config
from config import (SAMPLE_RATE, N_MFCC, UPLOAD_FOLDER, MAX_CONTENT_LENGTH,
                    DNN_MODEL_PATH, DNN_SCALER_PATH, XGBOOST_MODEL_PATH,
                    FLASK_DEBUG, FLASK_PORT)

# Import feature extractor
from src.feature_extraction import AudioFeatureExtractor

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# CORS
@app.after_request
def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ============================================================================
# LOAD MODELS
# ============================================================================

dnn_model = None
dnn_scaler = None
xgb_model = None
feature_extractor = None

def load_models():
    global dnn_model, dnn_scaler, xgb_model, feature_extractor
    
    print("\n" + "="*70)
    print("LOADING MODELS")
    print("="*70)
    
    try:
        print("[1/4] Loading DNN model...")
        dnn_model = tf.keras.models.load_model(str(DNN_MODEL_PATH))
        print("✅ DNN loaded")
        
        print("[2/4] Loading DNN scaler...")
        with open(str(DNN_SCALER_PATH), 'rb') as f:
            dnn_scaler = pickle.load(f)
        print("✅ Scaler loaded")
        
        print("[3/4] Loading XGBoost model...")
        xgb_model = XGBClassifier()
        xgb_model.load_model(str(XGBOOST_MODEL_PATH))
        print("✅ XGBoost loaded")
        
        print("[4/4] Initializing feature extractor...")
        feature_extractor = AudioFeatureExtractor(sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        print("✅ Feature extractor initialized")
        
        print("\n" + "="*70)
        print("✅ ALL MODELS LOADED SUCCESSFULLY")
        print("="*70 + "\n")
        
        return True
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        traceback.print_exc()
        return False

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET', 'POST', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return '', 204
    
    models_ready = all([dnn_model, dnn_scaler, xgb_model, feature_extractor])
    return jsonify({'status': 'healthy', 'models_loaded': models_ready}), 200

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    
    print("\n" + "="*70)
    print("NEW PREDICTION REQUEST")
    print("="*70)
    
    try:
        # Check if file exists
        if 'file' not in request.files:
            print("❌ No file in request")
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if not file or not file.filename:
            print("❌ No filename")
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Get filename as string
        filename = file.filename
        print(f"📄 Filename: {filename}")
        print(f"📄 Filename type: {type(filename)}")
        
        # Convert to string if needed
        if not isinstance(filename, str):
            filename = str(filename)
        
        print(f"📄 Filename after conversion: {filename} (type: {type(filename)})")
        
        # Validate extension - SIMPLE METHOD
        filename_lower = filename.lower()  # This is the line that might fail
        print(f"📄 Lowercase filename: {filename_lower}")
        
        valid_extensions = ['mp3', 'wav', 'flac', 'ogg', 'm4a']
        is_valid = False
        for ext in valid_extensions:
            if filename_lower.endswith('.' + ext):
                is_valid = True
                print(f"✅ Valid extension: .{ext}")
                break
        
        if not is_valid:
            print(f"❌ Invalid file extension")
            return jsonify({
                'success': False,
                'error': 'Invalid format. Supported: .mp3, .wav, .flac, .ogg, .m4a'
            }), 400
        
        # Check models
        if not all([dnn_model, dnn_scaler, xgb_model, feature_extractor]):
            print("❌ Models not loaded")
            return jsonify({'success': False, 'error': 'Models not loaded'}), 500
        
        # Save file
        from werkzeug.utils import secure_filename
        safe_name = secure_filename(filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        
        print(f"💾 Saving to: {filepath}")
        file.save(filepath)
        print("✅ File saved")
        
        try:
            # Extract features
            print("🎵 Extracting features...")
            features = feature_extractor.extract_features_from_file(filepath)
            
            if not features:
                print("❌ Feature extraction failed")
                return jsonify({'success': False, 'error': 'Feature extraction failed'}), 400
            
            print(f"✅ Extracted {len(features)} features")
            
            # Convert features to array - WITH ERROR HANDLING
            feature_values = []
            for key, value in features.items():
                # If value is array/list, take mean
                if isinstance(value, (list, np.ndarray)):
                    feature_values.append(float(np.mean(value)))
                else:
                    feature_values.append(float(value))
            
            X = np.array(feature_values).reshape(1, -1)
            print(f"📊 Feature matrix: {X.shape}")
            
            # DNN prediction
            print("🧠 DNN prediction...")
            X_scaled = dnn_scaler.transform(X)
            dnn_pred = float(dnn_model.predict(X_scaled, verbose=0)[0][0])
            print(f"   Score: {dnn_pred:.4f}")
            
            # XGBoost prediction
            print("🌲 XGBoost prediction...")
            xgb_pred = float(xgb_model.predict_proba(X)[0][1])
            print(f"   Score: {xgb_pred:.4f}")
            
            # Ensemble
            print("🤝 Ensemble...")
            ensemble_pred = (dnn_pred + xgb_pred) / 2
            is_phishing = ensemble_pred > 0.5
            confidence = max(ensemble_pred, 1 - ensemble_pred) * 100
            
            print(f"   Ensemble: {ensemble_pred:.4f}")
            print(f"   Classification: {'PHISHING' if is_phishing else 'LEGITIMATE'}")
            print(f"   Confidence: {confidence:.2f}%")
            
            # Response
            response = {
                'success': True,
                'classification': 'PHISHING DETECTED ⚠️' if is_phishing else '✅ LEGITIMATE CALL',
                'is_phishing': bool(is_phishing),
                'confidence': round(confidence, 2),
                'scores': {
                    'dnn': round(dnn_pred, 4),
                    'xgboost': round(xgb_pred, 4),
                    'ensemble': round(ensemble_pred, 4)
                }
            }
            
            print("\n✅ PREDICTION SUCCESSFUL\n")
            return jsonify(response), 200
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'}), 500
        
        finally:
            # Cleanup
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print("🗑️ Temp file deleted")
                except:
                    pass
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n🔐 VOICE PHISHING DETECTION SYSTEM\n")
    
    # Create upload folder
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Load models
    if not load_models():
        print("⚠️ Models failed to load!")
        print("Run training scripts first:")
        print("  python src/04_train_dnn.py")
        print("  python src/05_train_xgboost.py")
    
    # Start Flask
    print(f"\n🚀 Starting Flask on http://localhost:{FLASK_PORT}\n")
    app.run(debug=FLASK_DEBUG, port=FLASK_PORT, host='0.0.0.0', use_reloader=False)
