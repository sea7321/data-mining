import seaborn as seaborn
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import absolute
from numpy import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('payroll.csv')

# define predictor and response variables
X = df[['Base Pay', 'Overtime Pay']]
y = df['Total Pay']

# define cross-validation method to use
cv = KFold(n_splits=10, random_state=1, shuffle=True)

# build multiple linear regression model
model = LinearRegression()

# use k-fold CV to evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# view mean absolute error
print("Mean Absolute Error: {}".format(mean(absolute(scores))))

# view RMSE
print("RMSE: {}".format(sqrt(mean(absolute(scores)))))

# rng = np.random.default_rng(1234)
#
# # Generate data
# x = rng.uniform(0, 10, size=100)
# y = x + rng.normal(size=100)

x = df['Overtime Pay']
y = df['Base Pay']

seaborn.regplot(x='Overtime Pay', y='Base Pay', ci=None, data=df)

# plt.scatter(x, y)

# # Initialize layout
# fig, ax = plt.subplots(figsize=(9, 9))
#
# # Add scatter plot
# ax.scatter(x, y, edgecolors="k")
#
# # Fit linear regression via least-squares with numpy.polyfit
# # It returns a slope (b) and intercept (a)
# # deg=1 means linear fit (i.e. polynomial of degree 1)
# b, a = np.polyfit(x, y, deg=1)
#
# # Create sequence of 100 numbers from 0 to 100
# xseq = np.linspace(0, 10, num=100)
#
# # Plot regression line
# ax.plot(xseq, a + b * xseq, color="k", lw=2.5)

# plot the graph
plt.tight_layout()
plt.show()
