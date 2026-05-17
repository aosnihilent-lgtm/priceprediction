from sklearn.pipeline import Pipeline

from features.core_features import CoreFeatureBuilder
from features.historical_aggregator import HistoricalAggregator
from features.feature_selector import ModelFeatureSelector

def build_feature_pipeline(feature_set = "full"):
    feature_pipeline = Pipeline(
        steps=[
            ("core_features", CoreFeatureBuilder()),
            ("historical_features", HistoricalAggregator()),
            ("feature_selector", ModelFeatureSelector(feature_set = feature_set))
        ]
    )
    
    return feature_pipeline