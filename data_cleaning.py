"""
Created on Thu Oct 13 16:07:20 2022

@author: beatrizlourenco
"""

from os import path
from os.path import exists
import pandas as pd
import numpy as np

def get_data_of_gender( data, gender: str):

    return data[ (data['Gender'] == gender)]

def restructuring_portuguese_data(raw_data: pd.DataFrame) -> pd.DataFrame: # adds age, year and gender; joins male and female datasets
    """Restructuring portuguese data.
    
    Args:
        raw_data: dataframe with column names: 'Year', 'Age', 'mxf' (mortality rate females), 'mxm' (mortality rate males)
    
    Returns:
        Dataframe with columns 'Year', 'Age', 'Gender', 'mx' (mortality rate)
    """

    year_max= raw_data.Year.max()
    year_min= raw_data.Year.min()
    Year = []
    Age = []
    mx = []
    Gender = []

    for year in range(year_min, year_max+1):
        dt = raw_data[raw_data['Year'] == year]
        Year.extend((dt['Year'].to_list())*2)
        Age.extend((dt['Age'].to_list())*2)
        mxf = dt['mxf'].to_list()
        mx.extend(mxf)
        mxm = dt['mxm'].to_list()
        mx.extend(mxm)
        female = ['Female']*(len(mxf))
        Gender.extend(female)
        male = ['Male']*(len(mxm))
        Gender.extend(male)
    
    data = pd.DataFrame({ 'Year': Year, 'Age': Age, 'Gender': Gender, 'mx':mx}) 
    data["Age"] = pd.to_numeric(data["Age"], errors='coerce') #change data types
    data = data[data['Age'] < 100.0]
    data["mx"] = pd.to_numeric(data["mx"])
    
    return data


def override_missing_values(raw_data: pd.DataFrame, years_wd: int = 2) -> pd.DataFrame: #inputs values with the average
    """Inputes zero values (missing) with average along chosen year width. 

    Args:
        raw_data: Dataframe with columns 'Year', 'Age', 'Gender', 'mx' (mortality rate)
        years_wd: Band width of year for each to take the average around the missing value. Example: If we are missing a mortality rate for year 2016 of individuals at age 6, the value inputed will be an average between (year 2016 - years_wd) and (2016 + years_wd) for age 6.
    
    Returns:
        New DataFrame with inputed values.
    """
    
    years_wd = 2
    mv_rows = (raw_data[(raw_data['mx'] == 0)]).index
    
    for row in mv_rows:
        age = (raw_data.loc[row])['Age']
        current_year = raw_data.loc[row]['Year']
        gender =  raw_data.loc[row]['Gender']
        df = raw_data[ (raw_data['Year'] >= (current_year - years_wd) ) & (raw_data['Year'] <= (current_year+years_wd) ) & (raw_data['Age'] == age) & (raw_data['Gender'] == gender) ]
          
        raw_data.at[row, 'mx'] = df['mx'].mean()
        
    raw_data['logmx'] = np.log(raw_data['mx'])
    return raw_data

def split_data(data: pd.DataFrame, split_value: int) -> tuple:
    """ Splits data in two by a given year. The second half contains the split value.

    Args:
        data: pandas DataFrame with a 'Year' column.
        split_value: year for which we do the split.

    Returns:
        tuple with two pandas Dataframe
    """

    _1st_data = data[data['Year'] < split_value]
    _2nd_data = data[data['Year'] >= split_value]
    return _1st_data, _2nd_data

def preprocessing_dataPT(data: pd.DataFrame):

    """Restructures the portuguese dataset and inputs missing values. 
    
    Note: It's considered that the missing values are the zero ones.

    Args:
        raw_data: dataframe with column names: 'Year', 'Age', 'mxf' (mortality rate females), 'mxm' (mortality rate males)
    
    Returns:
        Dataframe with columns 'Year', 'Age', 'Gender', 'mx' (mortality rate)

    """
    dataPT = restructuring_portuguese_data(data)
    dataPT = override_missing_values(dataPT, years_wd = 2)
    return dataPT


def get_country_data(country: str, filedir: str = None):
    """Gets the clean and restructure country data.

    Args:
        country (str): can only be 'SW' (switzerland) or 'PT' (portugal).
        filedir (str): file directory to read. If None, will assume  'Dataset/dataPT.xlsx' if 'PT' and 'Dataset/dataSW.xlsx' if 'SW'.
    
    Returns:
        Clean dataFrame with country data.
    
    """

    filedir_dict = { 'PT': 'Dataset/dataPT.xlsx', 'SW': 'Dataset/dataSW.xlsx'}

    if country not in filedir_dict.keys():
        raise Exception("Invalid Country selected!")

    if not path.exists(filedir_dict.get(country)) or filedir != None:
        assert path.exists(filedir)
        if country == "PT":
            data = pd.read_csv(filedir, header=None, sep='\s+', names=["Year", "Age", "mxf", "mxm", 'mxTotal'])
            data = preprocessing_dataPT(data)
            data.to_excel(filedir_dict.get(country), index=False)
    
        elif country == 'SW':
            data = pd.DataFrame(pd.read_excel(filedir))
            data.to_excel(filedir_dict.get(country), index=False)
        
        return data

    else:
        return pd.DataFrame(pd.read_excel(filedir_dict.get(country)))

        






