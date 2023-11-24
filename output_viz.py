from matplotlib.ticker import LinearLocator
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pandas as pd


def mortalityrates3dviz(df_pred_test, name = '3DplotPredTransformer.pdf'):

    sns.set(style = "darkgrid")
    

    df_pred_test_bg=df_pred_test.groupby(['Year','Age']).mean()

    Year = []
    Age=[]
    for index in list(((df_pred_test.groupby(['Year','Age']).mean()).index).values):
        Year.append(index[0])
        Age.append(index[1])

    fig = plt.figure()

    ax = fig.add_subplot(111, projection = '3d')

    x = Age
    y = list(map(int, Year))
    z = df_pred_test_bg['logmx']
    plt.xticks(rotation=90)
    ax.set_xlabel("Age")
    ax.set_ylabel("Year")
    ax.set_zlabel("logmx", rotation=20)
    ax.plot_trisurf(x, y, z, cmap=cm.jet)
    ax.xaxis.set_major_locator(MaxNLocator(10))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.view_init(20, 360-90)
    plt.savefig(name)

    plt.show()

def bland_altman_plot(test, pred, gender, name = 'BlandAltmanFemale.pdf'):
    m1 = np.array((test[test['Gender']==gender])['mx'].to_list())
    m2 = np.array((pred[pred['Gender']==gender])['mx'].to_list())

    f, ax = plt.subplots(1)
    sm.graphics.mean_diff_plot(m1, m2, ax = ax)
    plt.xlabel(fontsize = 20, fontweight='bold')
    plt.ylabel(fontsize = 20, fontweight='bold')
    plt.savefig(name)
    plt.show()


def heatmap(pred, gender, name = 'heatmatFemalePredict.pdf', lim = [-11, 0]):
    pred_gender = pred[pred['Gender'] == gender]
    pred_test_female_mat = pd.crosstab(pred_gender['Year'], pred_gender['Age'], pred_gender['logmx'],aggfunc='sum')

    fig1, ax1 = plt.subplots(figsize=(10,10))
    ax1.grid(False)
    im1= ax1.imshow((pred_test_female_mat.to_numpy()), cmap='tab20c', interpolation='nearest',vmin = lim[0],vmax = lim[1]) #cmap choices: tab20c, prism, flag
    fig1.colorbar(im1, ticks = lim,
                fraction=0.03, 
                pad=0.04
                ).ax.tick_params(labelsize=16)
    locs, labels = plt.yticks()
    labels = [int(item)+2000 for item in locs]
    plt.yticks(locs, labels, fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.ylim([0,20])
    plt.xlim([0,99])
    plt.xlabel("Age", fontsize = 16, fontweight='bold')
    plt.ylabel("Year", fontsize = 16, fontweight='bold')
    plt.savefig(name)

    plt.show() 

