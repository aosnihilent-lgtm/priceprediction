import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from datetime import datetime
from dotenv import load_dotenv
from Trainning.Quote_Price_Trainning import train_quote_models
from Trainning.POP_Price_Trainning import train_pop_models
from utils.model_loader import load_model
from pricing.price_engine import PriceEngine
from utils.config_loader import load_config

load_dotenv()

app = Flask(__name__)
CORS(app)

# ======================================
# 🔥 LOAD MODELS ONCE (IMPORTANT)
# ======================================
# baseline_model = joblib.load("models/baseline_model.pkl")
# elasticity_model = joblib.load("models/elasticity_model.pkl")
# encoding_maps = joblib.load("models/baseline_encoders.pkl")
# baseline_features = joblib.load("models/baseline_features.pkl")

model = load_model()
config = load_config()
engine = PriceEngine(model, config)

# ======================================
# HOME HTML
# ======================================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# ======================================
# PRICE RECOMMENDATION API
# ======================================
@app.route("/price-recommendation", methods=["POST"])
def price_recommendation():

    try:
        request_json = request.get_json()
    
        # Fix column name mismatch
        if "MATGRP1" in request_json:
            request_json["MATGROUP1"] = request_json["MATGRP1"]

        # Feature Engineering (must match training)
        request_json["log_quantity"] = np.log1p(float(request_json["QUANTITY"]))

        # If you don't send VALIDFROM / VALIDTO, set default validity - not using these fields for now
        request_json["validity_days"] = 30

        # If CREATEDON not provided, use current date
        
        now = datetime.now()
        request_json["created_month"] = now.month
        request_json["created_year"] = now.year
        request_json["CREATEDON"] = now

        if not request_json:
            return jsonify({
                "status": "error",
                "message": "Invalid JSON input"
            }), 400


        # quote = request_json.dict()
        df = pd.DataFrame([request_json])
        result = engine.recommend_price(df)
        # return result

        # Create baseline dataframe with correct column order
        # baseline_df = pd.DataFrame([request_json])[baseline_features]

        # result = price_recommendation_with_quote(
        #     baseline_df,
        #     request_json,
        #     baseline_model,
        #     elasticity_model,
        #     encoding_maps,
        #     baseline_features
        # )

        return jsonify({
            "status": "success",
            "data": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ======================================
# TRAIN MODEL API
# ======================================
# new trainning model
@app.route("/training", methods=["GET"])
def train_models():

    try:
        result_quote = train_quote_models()
        #result_pop = train_pop_models()
        result = result_quote #+ result_pop
        return jsonify({
            "status": "success",
            "message": "Model trained successfully",
            "result": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ======================================
# RUN APP
# ======================================
if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)