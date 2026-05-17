from pricing.price_simulator import PriceSimulator
from pricing.revenue_optimizer import RevenueOptimizer

class PriceEngine:
    def __init__(self, model_pipeline, config) -> None:
        self.simulator = PriceSimulator(model_pipeline)
        self.optimizer = RevenueOptimizer()
        self.config = config
        
    def compute_price_bounds(self, quote):
        cfg = self.config["price_margins"]
        list_price = quote['STDINVOICEPRICE'].iloc[0]
        cost = quote['PERUNITCOST'].iloc[0]
        
        # Using Margin -> But Price in O/P of min, max and Best sometimes are same
        # min_price = cost * (1+ cfg['min_margin'])
        # max_price = cost * (1+ cfg['max_margin'])
        
        min_price = list_price * cfg['min_ratio']
        max_price = list_price * cfg['max_ratio']
        
        
        return min_price, max_price
    
    def predict_probability(self, quote, price):
        df = quote.copy()
        df["PERUNITQUOTEPRICE"] = price
        prob = self.simulator.model.predict_proba(df)[:,1][0]
        return prob
        
    def recommend_price(self, quote):
        
        min_price, max_price = self.compute_price_bounds(quote)
        
        # Probability at quoted price
        quote_prob = self.simulator.model.predict_proba(quote)[:, 1][0]
        
        simulation_df = self.simulator.simulate_prices(
            quote,
            min_price=min_price,
            max_price=max_price
        )
        
        pricing_cfg = self.config['pricing']
        
        
        # Best Price
        best_price, best_prob = self.optimizer.find_best_price(simulation_df)
        
        # Lowest Price for Highest Win Probability
        min_price_rec, min_proba = self.optimizer.find_probability_price(
            simulation_df,
            pricing_cfg['min_price_probability']
            )
        
        # # Price for mid probability
        # mid_price_rec, mid_prob = self.optimizer.find_probability_price(
        #     simulation_df,
        #     pricing_cfg['mid_price_probability']
        # )
        
        # # Max Price
        # max_price_rec, max_prob = self.optimizer.find_probability_price(
        #     simulation_df,
        #     pricing_cfg['max_price_probability']
        # )
        
        # Mid price = midpoint between min and best
        mid_price_rec = (min_price_rec + best_price) / 2
        mid_prob = self.predict_probability(quote, mid_price_rec)

        # Max price = stretch above optimal
        max_price_rec = min(best_price * 1.05, max_price)
        max_prob = self.predict_probability(quote, max_price_rec)


        status_cfg = self.config['probability_status']
        status= (
            "High Probability" if quote_prob > status_cfg['high']
            else "Medium Probability" if quote_prob > status_cfg['medium']
            else "Low Probability"
        )
        
        return {
            
                "Min_Price" : round(min_price_rec, 0),
                "Min_Price_Probability": round(min_proba * 100, 2),
                "Mid_Price": round(mid_price_rec, 0),
                "Mid_Price_Probability": round(mid_prob * 100, 2),
                "Max_Price": round(max_price_rec, 0),
                "Max_Price_Probability": round(max_prob * 100,2),
                "Best_Price": round(best_price, 0),
                "Best_Price_Probability": round(best_prob * 100, 2),
                "Win_Probability": round(quote_prob * 100, 2),
                "Win_Probability_Status": status
                
            
        }