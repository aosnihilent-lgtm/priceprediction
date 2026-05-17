import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def calibrate_model(base_model, X_val, y_val):
    """
    Fits isotonic calibration on validation data.
    Returns calibrated model ready for inference.
    """
    frozen_model = FrozenEstimator(base_model)
    calibrated = CalibratedClassifierCV(
        frozen_model,
        method='isotonic'
    )
    calibrated.fit(X_val,y_val)
    return calibrated


def calibration_table(y_true, proba, bins=10):
    df = pd.DataFrame({
        'y' : y_true,
        'p' : proba
    })
    df['bin'] = pd.qcut(df['p'], bins, duplicates='drop')
    
    table = df.groupby("bin").agg(
        avg_pred = ('p', 'mean'),
        actual_rate = ('y', 'mean'),
        count = ('y', 'size')
        )
    table['gap'] = table["avg_pred"] - table['actual_rate']
    print(table)
    print("\nMean avg calibration error:", table['gap'].abs().mean())
    return table['gap'].abs().mean()

