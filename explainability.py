
import torch
import pandas as pd

from torch import tensor 
import mortalityRateTransformer as mrt
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, IntegratedGradients, GradientShap
import matplotlib.pyplot as plt
import preprocessing_transformer as pt
import numpy as np

class ExplainableMortalityRateTransformer():
    """Wrapper for Captum framework"""
    
    def __init__(self, name:str, model: mrt.MortalityRateTransformer):
        self._name = name
        self._model = model
    
    
    def forward_func(self,xe: tensor, xd: tensor, ind: tensor):
        """
            Wrapper around prediction method of model
        """
        pred = self._model(xe, xd, ind)
        
        return pred[:, -1,:].squeeze(1)
        
    def visualize(self, attributes: list, dim_to_visualize:int,indexes:list = None):
        """
            Visualization method.
            Takes list of inputs and correspondent attributs for them to visualize in a barplot
        """
        attr_sum = attributes.sum(dim_to_visualize) 
        
        attr = attr_sum / torch.norm(attr_sum)
        
        if indexes is None:
            index_size = len(attr.numpy()[0])
            indexes = list(map(str, np.arange(index_size)))


        a = pd.Series(attr.numpy()[0], 
                         index = indexes)
        print(a)
        a.plot.barh()
                      
    def explain(self, inputs: tuple, dim_to_explain: int = 1, batched_input: bool = True):
        """
            Main entry method. Passes input through series of transformations and through the model. 
            Calls visualization method.
        """
        
        int_grad = GradientShap(self.forward_func)
        if batched_input:
            inputs = pt.unbatchify(inputs)
        
        baselines = self.generate_baseline(inputs)
        (xe_atr, xd_atr, ind_atr) = int_grad.attribute(inputs = inputs, baselines = baselines)
        
        self.visualize( xe_atr, dim_to_explain)
        #self.visualize( xd_atr, dim_to_explain)
        #self.visualize( ind_atr, dim_to_explain)

    def generate_baseline(self, inputs: tuple) -> tensor:
        """
            Convenience method for generation of baseline vector as tuple of torch tensors zeros
        """        
        return tuple( map(lambda x: torch.zeros(size = x.size()) if x is not None else None, inputs))
    
