import pandas as pd
import joblib
from Library.db import create_hana_connection, load_table
from sklearn.pipeline import Pipeline
from models.feature_building_pipeline import build_feature_pipeline
from models.model_training import train_lgbm_model


def structural_cleaning(df):
    df = df.copy()
    print("Lenth of input_file ", len(df))
    # Filter Quote < Cost (Selling Below Cost price)
    df = df[df.PERUNITQUOTEPRICE >= df.PERUNITCOST]
    print("Lenth of input_file after Filter Quote < Cost (Selling Below Cost price)", len(df))

    df['CREATEDON'] = pd.to_datetime(df['CREATEDON'])
    # Filter negative or zero price
    df = df[
        (df.PERUNITQUOTEPRICE > 0)
        & (df.STDINVOICEPRICE > 0)
        & (df.PERUNITCOST > 0)
                        ]
    print("Lenth of input_file after Filter negative or zero price", len(df))

    return df


def train_quote_models():

    print("🚀 Starting Quote Training Pipeline...")

    # ==============================
    # 1️⃣ LOAD DATA FROM HANA
    # ==============================
    conn = create_hana_connection()

    df = load_table(
        conn,
        table_name="QUOTE_ML_DATA",
        schema_name="IDEAL_SMART_QUOTE_HDI_IDEAL_SMART_QUOTE_DB_DEPLOYER_1"
    )

    conn.close()

    print("✅ Data Loaded:", df.shape)

    train = df[df["WINSTATUS"].isin(["WIN","LOSS"])].copy()
    train["target"] = (train["WINSTATUS"] == "WIN").astype(int)

    train = structural_cleaning(train)
    
    # -----------------------------
    # Load data
    # -----------------------------
    #train = pd.read_parquet("data/processed/train.parquet")
    # val = pd.read_parquet("data/processed/val.parquet")
    # test = pd.read_parquet("data/processed/test.parquet")

    y_train = train["target"]
    # y_val = val["target"]
    # y_test = test["target"]
    print(train[:10])
    # -----------------------------
    # Build features
    # -----------------------------
   
    feature_pipeline = build_feature_pipeline(feature_set="price_only")

    X_train = feature_pipeline.fit_transform(train, y_train)
    # X_val = feature_pipeline.transform(val)
    # X_test = feature_pipeline.transform(test)
    print("🚀 77...", X_train)

    # -----------------------------
    # Train base model
    # -----------------------------
    model = train_lgbm_model(
        X_train,
        y_train,
        feature_pipeline.named_steps["feature_selector"].feature_columns
    )

    # -----------------------------
    # Build production pipeline
    # -----------------------------
    pipeline = Pipeline([
        ("features", feature_pipeline),
        ("model", model)
    ])

    # -----------------------------
    # Save artifact
    # -----------------------------
    joblib.dump(pipeline, "artifacts/quote_price_model.joblib")

    print("Quote Model training complete.")
    return "Quote Model training complete."