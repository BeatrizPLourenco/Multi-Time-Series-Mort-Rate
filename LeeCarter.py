import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import data_cleaning as dtclean
import cvxpy as cp
from sklearn.metrics import mean_squared_error
np.random.seed(1)


class LeeCarter:

    def __init__(self, gender: str):
        self.gender = gender
        self.ax = np.array([])
        self.bx = np.array([])
        self.kt = np.array([])

    def fit(self, train_data: pd.DataFrame):
    
        train_data_by_gender = train_data[ (train_data['Gender'] == self.gender)]

        ## Construction of Centered log-Mortality Matrix
        # Get mean values for log mortality for each age 
        mean_logmx_by_age_dict = train_data_by_gender.groupby(by='Age').agg('mean')['logmx'].to_dict() 
        train_data_by_gender['ax'] = train_data_by_gender['Age'].map(mean_logmx_by_age_dict)
        train_data_by_gender['mx_adj'] = train_data_by_gender['logmx'] - train_data_by_gender['ax']

        
        rates_mat = pd.pivot_table(train_data_by_gender, index='Age', columns='Year', values='mx_adj', aggfunc='sum')

        # Singular value decomposition (SVD)
        U, d, V = np.linalg.svd(rates_mat, full_matrices=False )
        V=V.transpose()
        ax = np.array(list(mean_logmx_by_age_dict.values()))
        bx = U[:,0]*d[0]
        kt = V[:,0]

        c1=kt.mean()
        c2=bx.sum()

        # Average log-mortality rate at age x (Lee-Carter formula)
        self.ax = ax + c1*bx
        # Rate of time for a person aged x (Lee-Carter formula)
        self.bx = bx / c2
        # Time indexes for the annual mortality rates chage (Lee-Carter formula)
        self.kt = (kt-c1)*c2

    
    def kt_forecast(self, num_time_steps: int): #returns kt forecasting
        arima = ARIMA(endog = self.kt,  trend = "t", order=(0, 1, 0))
        fit_arima = arima.fit()
        kt_forecast = fit_arima.forecast(steps = num_time_steps) if num_time_steps != 0 else []

        return kt_forecast
    
    
    def predict(self, num_time_steps: int, return_only_prediction: bool = False):
        new_kt = self.kt_forecast(num_time_steps)

        if return_only_prediction:
            kt_to_output = new_kt
        else:
            kt_to_output = np.append(self.kt, new_kt)

        fitted = np.exp(self.ax[:,None] + (self.bx[:,None] @ kt_to_output[None,:]))
        df = pd.melt(pd.DataFrame(fitted))

        output = {'mx': df.value, 'kt_forecast': new_kt}
        return output
    
    def map_mx_to_kt(self, data: pd.DataFrame):
        year_range = data['Year'].unique()

        data_by_gender = data[data['Gender'] == self.gender]

        data_by_gender['ax'] = data_by_gender['Age'].map(pd.DataFrame(self.ax).to_dict()[0])
        data_by_gender['mx_adj'] = data_by_gender['logmx'] - data_by_gender['ax']
        mat=pd.pivot_table(data_by_gender, index='Age', columns='Year', values='mx_adj', aggfunc='sum')

        new_kt = []
        for year in year_range:
            n = 1 #only one calendar year
            A = - self.bx
            b = - np.array(mat[year])

            # Define and solve the CVXPY problem.
            x = cp.Variable(n)
            cost = cp.sum_squares(A * x - b)
            prob = cp.Problem(cp.Minimize(cost))
            prob.solve()
            new_kt.extend(list(x.value))
        return new_kt




def viz_kt_graph(kt: np.array, kt_forecasts: dict, x_axis_train_values: np.array, x_axis_val_values: np.array, path_to_save_fig: str == None):
        # Plot Settings
        plt.rcParams['figure.figsize']=(10,10)
        plt.rcParams.update({'mathtext.default':  'regular' })
        plt.xlabel('Calendar year',fontsize = 20, fontweight='bold')
        plt.ylabel('Values $k_{t}$',fontsize = 20, fontweight='bold')
        plt.yticks(fontsize = 18)
        plt.xticks(fontsize = 18)
        #plt.title('Estimated process $k_{t}$', fontsize = 20, fontweight='bold')

        #Scatter Plots
        plt.scatter(x = x_axis_train_values, y = kt, c = 'b')
        for id, kt_forecast in kt_forecasts.items():
            c, name = id
            plt.scatter(x = x_axis_val_values, y = kt_forecast, c = c, label = name)

        
        if path_to_save_fig is not None:
            plt.savefig(path_to_save_fig)
        plt.show()

    

if __name__ == '__main__':
    country = "PT"
    split_value = 2000
    split_value1 = 1993
    split_value2 = 2006
    gender = 'Female'
    data = dtclean.get_country_data(country)
    data_by_gender = dtclean.get_data_of_gender(data, gender)
    
    training_data, validation_test_data  = dtclean.split_data(data_by_gender, split_value2)
    #validation_data, test_data  = dtclean.split_data(validation_test_data, split_value2)
    year_range_train = training_data['Year'].unique()
    year_range_val = validation_test_data['Year'].unique()


    leecarter = LeeCarter(gender)
    leecarter.fit(training_data)
    prediction = leecarter.predict(num_time_steps = len(year_range_val), return_only_prediction = True)
    pred_mx = prediction['mx']
    LC_new_kt = prediction['kt_forecast']
    validation_test_kt = leecarter.map_mx_to_kt(data = validation_test_data)
    train_prediction = leecarter.predict(num_time_steps = 0, return_only_prediction = False)
    train_pred_mx = train_prediction['mx']

    kt_forecasts = { ('r', 'Lee Carter') : LC_new_kt} #,('b', None) : validation_test_kt}

    # Evaluation 
    mse_train = mean_squared_error(training_data['mx'], train_pred_mx )*10**4
    mse_val = mean_squared_error(validation_test_data['mx'], pred_mx )*10**4
    print(f'RMSE during training ({gender}) {round(mse_train, 4)}')
    print(f'RMSE during testing ({gender}) {round(mse_val, 4)}')
    from datetime import datetime

    viz_kt_graph(kt = leecarter.kt, kt_forecasts = kt_forecasts, x_axis_train_values = year_range_train, x_axis_val_values = year_range_val, path_to_save_fig = f'leecarter_{gender}_{country}_' + str(datetime.now()) + '.pdf')
    
    
    print('end!')


