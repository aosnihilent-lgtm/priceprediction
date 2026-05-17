import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class HistoricalAggregator(BaseEstimator, TransformerMixin):
    """
    Learns behavioural statistics from training data and maps them to new rows without leakage.
    """
    def __init__(self, zipcode_digits = 3):
        self.zipcode_digits = zipcode_digits
        
    def fit(self, X, y = None):
        df = X.copy()
        df['target'] = y
        
         #------------- Global Features -------------
        self.global_winrate = df['target'].mean()
        self.global_price_ratio = df['price_ratio'].mean()
        
        #------------- Customer -------------
        self.customer_stats = (
            df.groupby('SOLDTO')
            .agg(
                cus_winrate_hist = ("target", "mean"),
                cus_avg_price_ratio_hist = ('price_ratio', 'mean')
            )
        )
        
      
        
        #------------- Product -------------
        self.product_stats = (
            df.groupby("MATERIALNO")
            .agg(
                prod_winrate_hist = ("target", 'mean'),
                prod_avg_price_ratio_hist = ("price_ratio", "mean")
            )
        )
          #------------- Product PRODUCTHIERARCHY-------------darshan 15/5/26
        self.product_hierarchy_stats = (
            df.groupby("PRODUCTHIERARCHY")
            .agg(
                prod_hierarchy_winrate_hist = ("target", 'mean'),
                prod_hierarchy_avg_price_ratio_hist = ("price_ratio", "mean")
            )
        )
        #------------- SALES office------------- darshan 15/5/26
        self.sales_office_stats = (
            df.groupby('SALESOFFICE')
            .agg(
                sales_office_win_rate_hist = ("target", 'mean'),
                sales_office_avg_price_ratio_hist = ("price_ratio", 'mean')
            )
        )
        
        #------------- SALES ORG -------------
        self.org_stats = (
            df.groupby('SALESORG')
            .agg(
                org_win_rate_hist = ("target", 'mean'),
                org_avg_price_ratio_hist = ("price_ratio", 'mean')
            )
        )
        
        #------------- REGION -------------
        df["REGION"] = df["ZIPCODE"].astype(str).str[:self.zipcode_digits]
        
        self.region_stats = (
            df.groupby("REGION")
            .agg(
                region_winrate_hist = ("target", 'mean')
            )
        )
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Merge Stats
        df = df.merge(self.customer_stats, on = "SOLDTO", how = 'left')
        df = df.merge(self.product_stats, on = 'MATERIALNO', how = 'left')
        df = df.merge(self.org_stats, on = 'SALESORG', how = 'left')
        df = df.merge(self.product_hierarchy_stats, on = 'PRODUCTHIERARCHY', how = 'left')   #--darshan 15/5/26
        df = df.merge(self.sales_office_stats, on = 'SALESOFFICE', how = 'left')  #--darshan 15/5/26

        df['REGION'] = df['ZIPCODE'].astype(str).str[:self.zipcode_digits]
        df = df.merge(self.region_stats, on = 'REGION', how = 'left')
        
        # Fallback for unseen entities
        fill_cols = [
            'cus_winrate_hist', 'prod_winrate_hist', 'org_win_rate_hist', 'region_winrate_hist','prod_hierarchy_winrate_hist','sales_office_win_rate_hist' 
        ]
        
        for c in fill_cols:
            df[c] = df[c].fillna(self.global_winrate)
        
        ratio_cols = [
            'cus_avg_price_ratio_hist', 'prod_avg_price_ratio_hist', 'org_avg_price_ratio_hist',  'prod_hierarchy_avg_price_ratio_hist', 'sales_office_avg_price_ratio_hist'
        ]
        for c in ratio_cols:
            df[c] = df[c].fillna(self.global_price_ratio)
            
            
                #------------Price Gap for Customer -----------------
        df['customer_price_gap'] = df['price_ratio'] - df['cus_avg_price_ratio_hist']
        
            
        return df