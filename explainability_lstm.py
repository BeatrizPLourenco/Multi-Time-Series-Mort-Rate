
import numpy as np
import shap
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import seaborn as sns

class ModelWrapper:
  def __init__(self, model, T=8, tau0=5):
      self.model = model
      self.T = T
      self.tau0 = tau0

  def forward(self, inputs):
      num_samples = inputs.shape[0]
      # Split the combined input into separate inputs
      gender = inputs[:, -1]
      inputs = inputs[:, :-1].reshape(num_samples, self.T, self.tau0)

      # Perform the model prediction using the separate inputs
      predictions = self.model.predict([inputs, gender])  # Adjust according to your model's prediction function

      return predictions

def flat_input(input):
  mx, gender = input
  samples_num = mx.shape[0]
  feature_size = mx.shape[1] * mx.shape[2]
  mx = mx.reshape(samples_num, feature_size)
  gender = gender.reshape(samples_num, 1)
  return np.append(mx, gender, axis=1)

def feature_names(beg_year, middle_age,gender, T=8, tau0=5):
  delta = int((tau0 -1)/2)
  feature_names = []
  for year in range(beg_year, beg_year+T):
    for age in range(middle_age - delta, middle_age +delta +1):
      age = max(age, 0)
      age = min(age, 99)
      feature_names.append(f"({year},{age})")
  feature_names.append(f'{gender}')
  return feature_names

def get_instance(data, instance_idx):
  mx, gender = data
  return [mx[instance_idx:instance_idx+1], gender[instance_idx:instance_idx+1]]


def shap_barplot(shap_values, reshaped_instance_with_gender, feature_names, filepath):

  plt.figure(figsize=(8,8))
  summary_fig = shap.summary_plot(shap_values, features=reshaped_instance_with_gender, feature_names = feature_names, show=False, plot_size=[8,8]) #Previsão de idade 2 ano 2010(Age, Year)
  # Adjust the fontsize of the plot
  fontsize = 14
  plt.gca().legend([])
  plt.gca().tick_params(axis='x', labelsize=fontsize)
  plt.gca().tick_params(axis='y', labelsize=fontsize)
  plt.gca().set_xlabel('Absolute Shapley values', fontsize=18)
  plt.gca().set_ylabel('Calendar year and Age', fontsize=18)
  plt.savefig(filepath, bbox_inches = 'tight')

  # Show the plot
  plt.show()

# Define the integrated gradients function
def integrated_gradients(model, inputs, baseline_inputs, num_steps=50):

    # Calculate the path integral step size
    alpha = np.arange(1, num_steps + 1) / num_steps

    # Compute the gradients along the path from baseline to the inputs
    interpolated_inputs = [
        [baseline_input + alpha[i] * (input_data - baseline_input) for i in range(num_steps)]
        for baseline_input, input_data in zip(baseline_inputs, inputs)
    ]
    interpolated_inputs = [
        tf.convert_to_tensor(interpolated_input, dtype=tf.float32)
        for interpolated_input in interpolated_inputs
    ]

    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        predictions = tf.exp(model(interpolated_inputs)*(-1))

    gradients = tape.gradient(predictions, interpolated_inputs)

    # Calculate the average gradients
    avg_gradients = [
        tf.reduce_mean(gradient, axis=0)
        for gradient in gradients
    ]

    # Compute the integrated gradients
    integrated_grads = [
        (input_data - baseline_input) * avg_gradient
        for input_data, baseline_input, avg_gradient in zip(inputs, baseline_inputs, avg_gradients)
    ]

    return integrated_grads

def explain_scores_heatmap(mat,  vmin, vmax, filepath,beg_year = None, middle_age = None, xticklabels = None, yticklabels = None, explain_method = 'Explanability Score', cmap='hot', T=8, tau0 = 5):
  delta = int((tau0-1)/2)
  fig = plt.figure(figsize=(6, 4))
  if xticklabels is None or yticklabels is None:
     xticklabels=range(beg_year,beg_year+T)
     yticklabels=[max(min(i,99),0) for i in range(middle_age - delta, middle_age +delta +1)]
     
  ax = sns.heatmap(mat, linewidth=0.5 ,vmin=vmin, vmax = vmax, cmap=cmap, xticklabels = xticklabels, yticklabels = yticklabels)
  
  # Adjust the fontsize of the plot
  fontsize = 14
  ax.tick_params(axis='x', labelsize=9)
  ax.tick_params(axis='y', labelsize=9)
  ax.set_xlabel('Calendar year', fontsize=18)
  ax.set_ylabel('Age', fontsize=18)
  plt.savefig(filepath, bbox_inches = 'tight')
  plt.show()

def explain_scores_per_age_plot(list_of_arrays,label=['Female', 'Male'],color=['r', 'b'], gender ='both', year = 2011, explain_method = 'ExplanabilityScore' ):
  plt.figure(figsize=(11,8))
  for i in range(len(list_of_arrays)):
    plt.plot(np.arange(0,100),list_of_arrays[i], color = color[i], label= label[i])
  plt.tick_params(axis='x', labelsize=16)
  plt.tick_params(axis='y', labelsize=16)
  plt.xlabel('Age',fontsize=18, weight='bold')
  plt.ylabel(explain_method,fontsize=18, weight='bold')
  plt.legend(loc='upper right', fontsize=18, frameon=True)
  plt.savefig(f'{gender}_{explain_method}_values_{year}.pdf')
  plt.show()
