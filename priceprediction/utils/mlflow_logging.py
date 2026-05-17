import os
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt


def log_feature_importance(model, feature_names, tmpdir):
    """Logs feature importance for tree models to MLflow"""

    if not hasattr(model, "feature_importances_"):
        return

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    path = os.path.join(tmpdir, "feature_importance.csv")
    importance.to_csv(path, index=False)

    mlflow.log_artifact(path, artifact_path="diagnostics")


def log_probability_range(model_pipeline, test_df):
    """
    Logs min/max/mean predicted probabilities.
    Useful for detecting collapsed probability distributions.
    """

    proba = model_pipeline.predict_proba(test_df)[:, 1]

    mlflow.log_metric("test_min_probability", float(proba.min()))
    mlflow.log_metric("test_max_probability", float(proba.max()))
    mlflow.log_metric("test_mean_probability", float(proba.mean()))

    return proba


def log_probability_distribution(proba, tmpdir):
    """
    Logs probability histogram.
    Very useful for detecting probability compression.
    """

    plt.figure()
    plt.hist(proba, bins=30)
    plt.xlabel("Predicted Win Probability")
    plt.ylabel("Frequency")
    plt.title("Probability Distribution")

    fig_path = os.path.join(tmpdir, "probability_distribution.png")
    plt.savefig(fig_path)
    plt.close()

    mlflow.log_artifact(fig_path, artifact_path="diagnostics")


def log_global_price_response(model_pipeline, raw_test_df, tmpdir, n_quotes=200):
    """
    Simulates price changes across multiple quotes to produce a stable elasticity curve.
    """

    sample_df = raw_test_df.sample(min(n_quotes, len(raw_test_df))).copy()

    base_price = sample_df["PERUNITQUOTEPRICE"].median()

    prices = np.linspace(base_price * 0.7, base_price * 1.3, 50)

    avg_probs = []

    for price in prices:

        sim_df = sample_df.copy()

        sim_df["PERUNITQUOTEPRICE"] = price

        proba = model_pipeline.predict_proba(sim_df)[:, 1]

        avg_probs.append(proba.mean())

    # Probability curve
    plt.figure()
    plt.plot(prices, avg_probs)
    plt.xlabel("Price")
    plt.ylabel("Average Win Probability")
    plt.title("Global Price Response Curve")

    fig_path = os.path.join(tmpdir, "price_probability_curve.png")
    plt.savefig(fig_path)
    plt.close()

    mlflow.log_artifact(fig_path, artifact_path="diagnostics")

    return prices, avg_probs


def log_expected_revenue_curve(prices, probs, tmpdir):
    """
    Logs expected revenue curve.
    """

    revenue = prices * probs

    plt.figure()
    plt.plot(prices, revenue)
    plt.xlabel("Price")
    plt.ylabel("Expected Revenue")
    plt.title("Price vs Expected Revenue")

    fig_path = os.path.join(tmpdir, "price_revenue_curve.png")
    plt.savefig(fig_path)
    plt.close()

    mlflow.log_artifact(fig_path, artifact_path="diagnostics")


def log_empirical_price_elasticity(X_test, y_test, feature_names, tmpdir):
    """
    Logs empirical win-rate vs price ratio bins.
    """

    df = pd.DataFrame(X_test, columns=feature_names)

    df["target"] = y_test.values

    if "price_ratio" not in df.columns:
        return

    df["price_ratio_bin"] = pd.qcut(df["price_ratio"], 20, duplicates="drop")

    elasticity_df = (
        df.groupby("price_ratio_bin")["target"]
        .mean()
        .reset_index()
    )

    plt.figure()
    plt.plot(elasticity_df["price_ratio_bin"].astype(str), elasticity_df["target"])
    plt.xticks(rotation=45)
    plt.xlabel("Price Ratio Bin")
    plt.ylabel("Win Rate")
    plt.title("Empirical Price Elasticity")

    fig_path = os.path.join(tmpdir, "empirical_price_elasticity.png")
    plt.savefig(fig_path)
    plt.close()

    mlflow.log_artifact(fig_path, artifact_path="diagnostics")


def log_all_diagnostics(model, pipeline, raw_test_df, X_test, y_test, feature_names, tmpdir):
    """
    Wrapper to log all experiment diagnostics.
    Keeps experiment script clean.
    """

    # Feature importance
    log_feature_importance(model, feature_names, tmpdir)

    # Probability range
    proba = log_probability_range(pipeline, raw_test_df)

    # Probability distribution
    log_probability_distribution(proba, tmpdir)

    # Global price response curve
    prices, probs = log_global_price_response(
        pipeline,
        raw_test_df,
        tmpdir
    )

    # Expected revenue curve
    log_expected_revenue_curve(prices, probs, tmpdir)

    # Empirical elasticity
    log_empirical_price_elasticity(
        X_test,
        y_test,
        feature_names,
        tmpdir
    )