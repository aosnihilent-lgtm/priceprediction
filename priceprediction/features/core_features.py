import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CoreFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Builds deterministic per-row features.
    No aggregation, no fitting, no leakage
    """
    def __init__(self):
        pass
    
    # Nothing to learn
    def fit(self, X, y = None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        df = X.copy()
        
        #------------- Price Features ------------- 
        df['price_ratio'] = df['PERUNITQUOTEPRICE'] / df['STDINVOICEPRICE']
        
        # Margin Percentage
        df['margin_pct'] = (
            (df['PERUNITQUOTEPRICE'] - df['PERUNITCOST']) / df['PERUNITQUOTEPRICE']
        )
        
        #------------- Quantity / Deal Size ------------- 
        df['log_qty']= np.log1p(df['QUANTITY']) # 1to2 1000 10001 
        df['deal_value'] = np.log1p(df["PERUNITQUOTEPRICE"]) * df['QUANTITY']
        
        #------------- Time Context -------------
        month = df['CREATEDON'].dt.month # 12 & 1
        df['month_sin'] = np.sin(2 * np.pi * month/ 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        #------------- Geography -------------
        df['is_cad_customer'] = (df['CURRENCY'] == "CAD").astype(int)
        
        # keep_cols = [
        #     'price_ratio',
        #     'margin_pct',
        #     'log_qty',
        #     'deal_value',
        #     'month_sin',
        #     'month_cos',
        #     'is_cad_customer'
        # ]
        return df