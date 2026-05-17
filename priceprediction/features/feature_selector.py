from sklearn.base import BaseEstimator, TransformerMixin

class ModelFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Finalizes the feture set used by the ML model.
    This becomes the contract between training & inference
    """
    def __init__(self, feature_set = "full"):
        self.feature_set = feature_set
        # ONLY place where model features are defined
        base_features = [
            #price effect
            "price_ratio",
            
            # deal economics
            "margin_pct",
            'log_qty',
            "deal_value",
            
            # Seasonality
            "month_sin",
            "month_cos",
            
            # Geography
            "is_cad_customer",
            
            # Customer Behaviour
            "cus_winrate_hist",
            "cus_avg_price_ratio_hist",
            
            # Product Behaviour
            "prod_winrate_hist",
            "prod_avg_price_ratio_hist",
            
            # Org Behavior
            "org_win_rate_hist",
            "org_avg_price_ratio_hist",
            
            # Sales Office Behavior -- darshan 15/5/26  
            "sales_office_win_rate_hist",
            "sales_office_avg_price_ratio_hist",

            # Product Hierarchy Behaviour
            "prod_hierarchy_winrate_hist"
            "prod_hierarchy_avg_price_ratio_hist",
            
            # Region Behavior
            "region_winrate_hist",
            
            #  # NEW FEATURE
            "customer_price_gap"
        ]
        if feature_set == "full":
            self.feature_columns = base_features
            
        elif feature_set == "no_winrate":
            self.feature_columns = [
                f for f in base_features
                if f not in [
                    'cus_winrate_hist',
                    'prod_winrate_hist',
                    'org_win_rate_hist',
                    'region_winrate_hist'
                ]
            ]
        elif feature_set == 'price_only':
            self.feature_columns = [
                'price_ratio',
                'log_qty',
                'month_sin',
                'month_cos',
                'is_cad_customer',
                'cus_avg_price_ratio_hist',
                'prod_avg_price_ratio_hist',
                'org_avg_price_ratio_hist',
                'sales_office_avg_price_ratio_hist',  #--darshan 15/5/26
                'prod_hierarchy_avg_price_ratio_hist',  #--darshan 15/5/26
                'customer_price_gap'
            ]
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required_features: {missing}")
        return X[self.feature_columns]