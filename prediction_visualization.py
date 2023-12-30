def kt_LSTM(gender, prediction):

  if gender == 'Male':
    mean_logmx = mean_logmx_by_age_dict_m
    bx = bx_male
  elif gender == 'Female':
    mean_logmx = mean_logmx_by_age_dict_f
    bx = bx_female

  prediction['ax'] = prediction.Age.map(mean_logmx)
  prediction['mx_adj'] = prediction.logmx - prediction.ax
  mat=pd.pivot_table(prediction, index='Age', columns='Year', values='mx_adj', aggfunc='sum')

  kt_LSTM=[]
  for year in range(2000,year_max+1):
    # Generate data.
    m = 100
    n = 1
    np.random.seed(1)
    A = -bx
    b = -np.array(mat[year])

    # Define and solve the CVXPY problem.
    x = cp.Variable(n)
    cost = cp.sum_squares(A * x - b)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()
    kt_LSTM.append(float(x.value))
  return kt_LSTM

kt_LSTM_bg_m = kt_LSTM('Male', prediction_bg_m)

kt_LSTM_bg_m = kt_LSTM('Male', prediction_bg_m)
kt_LSTM_og_m = kt_LSTM('Male', prediction_og_m)

kt_LSTM_bg_f = kt_LSTM('Female', prediction_bg_f)
kt_LSTM_og_f = kt_LSTM('Female', prediction_og_f)

kt_test_f = kt_LSTM('Female', test_data[test_data['Gender'] == 'Female'])
kt_test_m = kt_LSTM('Male', test_data[test_data['Gender'] == 'Male'])

sns.set(style = "darkgrid")

# Plot Male kt
plt.figure()
plt.rcParams['figure.figsize']=(20,10)
plt.rcParams.update({'mathtext.default':  'regular' })
t_forecast=np.arange(2000,year_max+1)
x=np.append(np.arange(year_min,2000), t_forecast)
y=np.append(kt_male, kt_forecast_m)
#forecast_col = np.append(np.zeros(len(kt_male)), np.ones(len(kt_forecast_m)))
plt.scatter(x=t_forecast, y=kt_LSTM_bg_m , c='g', label="LSTM both genders") # forecast for LSTM both genders for Males
plt.scatter(x=t_forecast, y=kt_LSTM_og_m , c='y', label="LSTM one gender") # forecast for LSTM one genders for Males
plt.scatter(x=t_forecast, y=kt_forecast_m, c='r',label="Lee Carter")
plt.scatter(x=np.arange(year_min,2000), y=kt_male, c='b')
plt.scatter(x=np.arange(2000,year_max+1), y=kt_test_m, c='b')
plt.legend(loc="lower left")
plt.xlabel('Calendar year',fontsize = 16, fontweight='bold')
plt.ylabel('Values $k_{t}$',fontsize = 16, fontweight='bold')
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.title('Estimated process $k_{t}$ for Males',fontsize = 18, fontweight='bold')
#plt.savefig('Plot-Models/LC-RNN-males.pdf')
plt.show()


# Plot Female kt
plt.figure()
plt.rcParams['figure.figsize']=(20,10)
plt.rcParams.update({'mathtext.default':  'regular' })
x=np.append(np.arange(year_min,2000), t_forecast)
y=np.append(kt_female, kt_forecast_f)
#forecast_col = np.append(np.zeros(len(kt_female)), np.ones(len(kt_forecast_f)))
plt.scatter(x=t_forecast, y=kt_LSTM_bg_f , c='g', label="LSTM both genders") # forecast for LSTM both genders for Females
plt.scatter(x=t_forecast, y=kt_LSTM_og_f , c='y',label="LSTM one gender") # forecast for LSTM one genders for Females
plt.scatter(x=np.arange(year_min,2000), y=kt_female, c='b')
plt.scatter(x=t_forecast, y=kt_forecast_f, c='r',label="Lee Carter")
plt.scatter(x=np.arange(2000,year_max+1), y=kt_test_f, c='b')

plt.legend(loc="lower left")
plt.xlabel('Calendar year',fontsize = 16, fontweight='bold')
plt.ylabel('Values $k_{t}$',fontsize = 16, fontweight='bold')
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.title('Estimated process $k_{t}$ for Females',fontsize = 18, fontweight='bold')
#plt.savefig('Plot-Models/LC-RNN-females.pdf')
plt.show()
