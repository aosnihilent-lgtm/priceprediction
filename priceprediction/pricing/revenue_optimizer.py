class RevenueOptimizer:
    
    def find_best_price(self, simulation_df):
        
        best = simulation_df.loc[
            simulation_df['revenue'].idxmax()
            ]
        return best['price'], best['probability']
    
    def find_probability_price(self, simulation_df, target_win_probability):
        # Find closet price for the desired win probability
        # idx = (simulation_df['probability'] - target_win_probability).abs().idxmin()
        # row = simulation_df.loc[idx]
        # return row['price'], row['probability']
        eligible = simulation_df[simulation_df["probability"] <= target_win_probability]
         # if never crosses threshold, return highest simulated price
        if len(eligible) == 0:
            row = simulation_df.iloc[-1]
        else:
            row = eligible.iloc[0]

        return row["price"], row["probability"]
