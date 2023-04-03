import pandas as pd
import numpy as np


def normalize_data():
    df = pd.read_csv('Telco.csv')
    my_data = []

    updated_tenure = []
    miny = min(df['tenure'])
    maxy = max(df['tenure'])
    for x in range(len(df['tenure'])):
        updated_tenure.append((df['tenure'][x] - miny) / (maxy - miny))
    my_data.append(updated_tenure)

    updated_monthly = []
    miny = min(df['MonthlyCharges'])
    maxy = max(df['MonthlyCharges'])
    for x in range(len(df['MonthlyCharges'])):
        updated_monthly.append((df['MonthlyCharges'][x] - miny) / (maxy - miny))
    my_data.append(updated_monthly)

    churn = []
    for i in df['Churn']:
        if i == 'Yes':
            churn.append(1)
        else:
            churn.append(0)
    my_data.append(churn)

    gender = []
    for i in df['gender']:
        if i == 'Female':
            gender.append(0)
        else:
            gender.append(1)
    my_data.append(gender)

    dependents = []
    for i in df['Dependents']:
        if i == 'Yes':
            dependents.append(1)
        else:
            dependents.append(0)
    my_data.append(dependents)

    partner = []
    for i in df['Partner']:
        if i == 'Yes':
            partner.append(1)
        else:
            partner.append(0)
    my_data.append(partner)
    return np.array(my_data)
