import pandas as pd
import numpy as np

class PriceSimulator:
    """
    Simulates win probability across candidate prices
    """
    def __init__(self, model_pipeline):
        self.model = model_pipeline
        
    def simulate_prices(self, quote_row, min_price, max_price, steps = 200):
        
        # Generate Candidate Prices
        prices = np.linspace(min_price, max_price, steps)
        
        # Repeate the quote row for all candidate prices
        df = pd.concat([quote_row] * steps, ignore_index=True)
        
        # Replace price column with simulated prices
        df['PERUNITQUOTEPRICE'] = prices
        
        # Predict probabilities for all rows at once
        proba = self.model.predict_proba(df)[:, 1]
        
        result = pd.DataFrame({
            'price' : prices,
            "probability" : proba
        })
        
        result['revenue'] = result['price'] * result['probability']
        
        return result