import joblib

def load_model(path = 'artifacts/quote_price_model.joblib'):
    return joblib.load(path)

def load_pop_model(path = 'artifacts/pop_price_model.joblib'):
    return joblib.load(path)