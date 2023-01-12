"""
https://towardsdatascience.com/explainable-ai-xai-with-shap-regression-problem-b2d63fdca670
"""

import shap
from torch import nn, Tensor

def explain_with_shap(model:nn.Module, input: Tensor, feature_names: list):
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input)
    shap.summary_plot(shap_values, input, feature_names = feature_names, plot_type="bar")
    shap.dependence_plot(5, shap_values, input, feature_names = feature_names)
    shap.decision_plot(explainer.expected_value[0], shap_values[0], feature_names = list(feature_names))

    return