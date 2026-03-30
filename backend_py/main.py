from flask import Flask, request, jsonify
from flask_cors import CORS # Penting agar Next.js bisa akses
import lightgbm as lgb
import pandas as pd
import os
import joblib
from functools import wraps

app = Flask(__name__)
CORS(app) # Mengizinkan request dari domain berbeda (localhost:3000)

# SEMENTARA UNTUK LOCALAN, nanti bisa pakai env variable atau config file
API_KEY = "jangkrik97"


MODEL_PATH_1 = 'hazard_model.txt'
LABEL_ENCODER_MODEL_1 = 'label_encoder.pkl'



# --- MIDDLEWARE / DECORATOR ---
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Mengambil key dari header 'x-api-key'
        user_key = request.headers.get('x-api-key')
        
        if not user_key or user_key != API_KEY:
            return jsonify({
                "status": "error", 
                "message": "Unauthorized: API Key salah atau tidak ada"
            }), 401
            
        return f(*args, **kwargs)
    return decorated_function

if os.path.exists(MODEL_PATH_1) and os.path.exists(LABEL_ENCODER_MODEL_1):
    # Menggunakan booster karena file berupa .txt export
    model_1 = lgb.Booster(model_file=MODEL_PATH_1)
    label_encoder = joblib.load(LABEL_ENCODER_MODEL_1)
    print("✅ Model LightGBM berhasil dimuat.")
else:
    print("❌ File model tidak ditemukan!")
    

grid_df = pd.read_csv('grid_df.csv')
def get_nearest_features(lat, lon, grid_df):
    distances = (
        (grid_df["lat"] - lat)**2 +
        (grid_df["lon"] - lon)**2
    )
    
    idx = distances.idxmin()
    row = grid_df.loc[[idx]]  # pakai [[ ]] biar tetap 2D
    
    return row[[
        # "freq",
        "max_mag",
        "avg_mag",
        "avg_depth",
        "gempa_in_radius_50",
        "density"
    ]]

@app.route('/predict', methods=['GET'])
@require_api_key
def predict():
    try:
        # 2. Ambil parameter dari request
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)

        if lat is None or lng is None:
            return jsonify({"error": "Parameter lat dan lng diperlukan"}), 400

        # fitur engineering
        nearest_features = get_nearest_features(lat, lng, grid_df)
        print(nearest_features)

        pred = model_1.predict(nearest_features)
        print(pred)
        prediction = pred.argmax(axis=1)
        label = label_encoder.inverse_transform(prediction)[0]
        
        print(label)

        return jsonify({
            "status": "success",
            "data": {
                "hazard_level": label,
                # "label": labels.get(prediction, "Unknown"),
                # "confidence": round(confidence, 4),
                "lat": lat, 
                "lng": lng
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Jalankan di port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)