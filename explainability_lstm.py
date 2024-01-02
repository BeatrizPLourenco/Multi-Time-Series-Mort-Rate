
import numpy as np
import shap
import matplotlib.pyplot as plt

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
