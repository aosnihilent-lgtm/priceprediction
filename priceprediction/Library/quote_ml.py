import numpy as np
import pandas as pd


def apply_target_encoding(df, encoding_maps):
    for col, mapping in encoding_maps.items():
        df[col] = df[col].map(mapping["means"]).fillna(mapping["global_mean"])
    return df


def price_recommendation_with_quote(
    sample_baseline_row,
    sample_original_row,
    baseline_model,
    elasticity_model,
    encoding_maps,
    baseline_features
):

    # ==============================
    # 1️⃣ APPLY TARGET ENCODING
    # ==============================
    sample_baseline_row = sample_baseline_row.copy()
    sample_baseline_row = apply_target_encoding(
        sample_baseline_row,
        encoding_maps
    )

    # Ensure correct feature order
    sample_baseline_row = sample_baseline_row[baseline_features]

    # IMPORTANT: No reshape, no .values
    baseline_prob = baseline_model.predict_proba(sample_baseline_row)[0][1]

    # ==============================
    # 2️⃣ ORIGINAL PRICE VALUES
    # ==============================
    cost = float(sample_original_row["PERUNITCOST"])
    invoice = float(sample_original_row["STDINVOICEPRICE"])
    quote_price = float(sample_original_row["PERUNITQUOTEPRICE"])

    # ==============================
    # 3️⃣ ELASTICITY CALCULATION
    # ==============================
    def compute_final_probability(price):
        price_to_invoice_ratio = price / invoice
        discount_pct = (invoice - price) / invoice
        margin_pct = (price - cost) / price
        price_to_cost_ratio = price / cost

        temp = pd.DataFrame([{
            "price_to_invoice_ratio": price_to_invoice_ratio,
            "discount_pct": discount_pct,
            "margin_pct": margin_pct,
            "price_to_cost_ratio": price_to_cost_ratio
        }])

        elasticity_prob = elasticity_model.predict_proba(temp)[0][1]
        return baseline_prob * elasticity_prob

    current_prob = compute_final_probability(quote_price)

    # ==============================
    # 4️⃣ PRICE SIMULATION
    # ==============================
    min_allowed_price = cost * 1.05
    max_allowed_price = invoice

    price_range = np.linspace(min_allowed_price, max_allowed_price, 60)

    results = []

    for price in price_range:
        final_prob = compute_final_probability(price)
        expected_revenue = price * final_prob
        results.append((price, final_prob, expected_revenue))

    results_df = pd.DataFrame(
        results,
        columns=["Price", "Win_Probability", "Expected_Revenue"]
    )

    min_price_row = results_df.loc[results_df["Win_Probability"].idxmax()]
    max_price_row = results_df.loc[results_df["Price"].idxmax()]
    mid_price_row = results_df.iloc[len(results_df)//2]
    best_price_row = results_df.loc[results_df["Expected_Revenue"].idxmax()]

    # ==============================
    # 5️⃣ LABEL FUNCTION
    # ==============================
    def label(prob):
        if prob >= 0.75:
            return "High Probability"
        elif prob >= 0.50:
            return "Medium Probability"
        else:
            return "Low Probability"

    # ==============================
    # 6️⃣ FINAL OUTPUT (UNCHANGED)
    # ==============================
    output = {
        "Current_Quote_Price": round(float(quote_price), 2),
        "Current_Quote_Probability": round(float(current_prob) * 100, 2),
        "Current_Quote_Status": label(current_prob),

        "Min_Price": round(float(min_price_row["Price"]), 2),
        "Min_Price_Probability": round(float(min_price_row["Win_Probability"]) * 100, 2),
        "Min_Price_Status": label(min_price_row["Win_Probability"]),

        "Mid_Price": round(float(mid_price_row["Price"]), 2),
        "Mid_Price_Probability": round(float(mid_price_row["Win_Probability"]) * 100, 2),
        "Mid_Price_Status": label(mid_price_row["Win_Probability"]),

        "Max_Price": round(float(max_price_row["Price"]), 2),
        "Max_Price_Probability": round(float(max_price_row["Win_Probability"]) * 100, 2),
        "Max_Price_Status": label(max_price_row["Win_Probability"]),

        "Best_Price": round(float(best_price_row["Price"]), 2),
        "Best_Price_Probability": round(float(best_price_row["Win_Probability"]) * 100, 2),
        "Best_Price_Status": label(best_price_row["Win_Probability"])
    }

    return output