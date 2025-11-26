import pandas as pd
import numpy as np
from sklearn.metrics import auc
from glum import TweedieDistribution
import typing

def evaluate_predict(predictions: pd.Series,
         actuals: pd.Series, 
         weight: pd.Series,
         Tweedie_power: float,
         model_type: str) -> pd.DataFrame:
    """ This function computes weighted bias,
    deviance, MSE and MAE for model predictions.
    Results are printed to a pandas DataFrame 
    object.

    Args:
        predictions (pd.Series): Model predictions
        actuals (pd.Series): Observed values
        weight (pd.Series): Sample weight

    Returns:
        pd.DataFrame: DataFrame containing prediction measures
    """
    # Bias: as deviation from the actual exposure adjusted mean.
    predictions_waverage = np.average(predictions, weights= weight)
    actuals_waverage = np.average(actuals, weights = weight)
    bias = predictions_waverage - actuals_waverage
    
    # Weighted MSE:sum of squared differences between observed and 
    # predicted values. 
    weighted_MSE = np.average((predictions-actuals)**2, weights = weight)
    
    # Weighted MAE: sum of differences between observed and 
    # predicted values, and weighted by passed weights.
    weighted_MAE = np.average(abs(predictions-actuals) , weights = weight)
    
    # Deviance (weighted):
    # Claims were modelled to follow Tweedie distribution. Hence we write the 
    # log likelihood to reflect this.
    weight_vals = weight.values
    TweedieDist = TweedieDistribution(Tweedie_power)
    deviance = TweedieDist.deviance(actuals, predictions, sample_weight=weight_vals) / np.sum(weight_vals)
    # Gini coefficient:
    y_true = np.asarray(actuals)
    y_pred = np.asarray(predictions)
    exposure = np.asarray(weight)
    
    ranking = np.argsort(y_pred) # returns the indices of sorted array
    ranked_exposure = exposure[ranking]
    ranked_actuals = y_true[ranking]
    # actuals:
    cumulated_actuals = np.cumsum(ranked_actuals * ranked_exposure)
    # normalize [0,1]
    cumulated_actuals /= cumulated_actuals[-1]
    # sample cumulated:
    cumulated_samples = np.linspace(0, 1, len(cumulated_actuals))
    # gini: auc calculates the area under the curve using trapezoid rule
    gini = 1 - 2 * auc(cumulated_samples, cumulated_actuals)
    gini = round(gini,3)
    
    # put everything in a nice df and report results
    results = pd.DataFrame({
        'Bias': [bias],
        'Weighted Mean Squared Error': [weighted_MSE],
        'Weighted Mean Absolute Error': [weighted_MAE],
        f'Deviance (Tweedie p={Tweedie_power})': [deviance],
        'Gini coefficient': [gini]        
        })
    results = results.style.set_caption(f"Model Evaluation Metrics {model_type}")\
                       .format("{:.3f}")
    
    return results

    
