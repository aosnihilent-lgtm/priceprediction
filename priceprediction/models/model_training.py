import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


def train_logistic_model(X_train, y_train, max_iter = 1000):
    model = LogisticRegression(
        max_iter= max_iter,
        n_jobs=None
    )
    
    model.fit(X_train, y_train)
    return model

def train_lgbm_model(X_train, y_train, feature_names):
    
    
    price_idx = feature_names.index('price_ratio')
    monotone_constraints = [0] * len(feature_names)
    monotone_constraints[price_idx] = -1
    gap_idx = feature_names.index("customer_price_gap")
    monotone_constraints[gap_idx] = -1
    
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=800,
        learning_rate=0.03,
        num_leaves= 64,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        monotone_constraints=monotone_constraints,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
    
    

def evaluate_model(model, X, y, name = 'dataset'):
    proba = model.predict_proba(X)[:,1]
    
    metrics = {
        "auc" : roc_auc_score(y, proba),
        "logloss": log_loss(y, proba),
        "brier" : brier_score_loss(y, proba)
    }
    print(f"\n{name} metrics: ")
    for k,v in metrics.items():
        print(f"{k}: {v:.5f}")
    
    return metrics