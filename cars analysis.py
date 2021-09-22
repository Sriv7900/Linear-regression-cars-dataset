import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


cars_df = pd.read_csv(r'C:\Users\Satya Srivastava\Downloads\Car details v3.csv')

#dropna rows from df
cars_df.dropna(inplace = True)

#function to convert mileage, engine, max power columns into an integer, and creating new column 'mileage_float'
def transform_to_float(x):
    return float(''.join(i for i in x if i.isdigit() or i == '.'))


cars_df['mileage_float'] = cars_df['mileage'].apply(lambda x: transform_to_float(x))
cars_df['engine_float'] = cars_df['engine'].apply(lambda x: transform_to_float(x))
cars_df['maxPower_float'] = cars_df['max_power'].apply(lambda x: transform_to_float(x))


#using the processed data for correlation matrix
columns = list(cars_df.columns)

numerical_variables = cars_df[['selling_price', 'km_driven', 'mileage_float', 'engine_float', 'maxPower_float', 'seats', 'year']]

corrMatrix = numerical_variables.corr()


#low correlation between seats and selling price, but this is likely because of there only being a few unique values
#for seats, potential mutlicollinearity between engine and max power, as well
sns.heatmap(corrMatrix, vmin = -1, vmax = 1, square = True, annot = True, cmap = "vlag")


numeric_cols = list(numerical_variables.columns)

for i in numeric_cols:
    if i != 'selling_price':
        sns.scatterplot(data = numerical_variables, x = i, y = 'selling_price')
        plt.show()
        plt.clf()


dummy_variables = ['seller_type', 'fuel', 'transmission', 'owner']

cars_df_dummies = cars_df

for i in dummy_variables:
    cars_df_dummies = cars_df_dummies.join(pd.get_dummies(cars_df[i]))

model_data = cars_df_dummies.drop(['mileage', 'engine', 'max_power', 'fuel', 'seller_type', 'torque', 'transmission', 'owner'], axis = 'columns')
X_vars = model_data.drop(['name', 'selling_price', 'mileage_float', 'engine_float'], axis = 'columns')
y_var = model_data[['selling_price']]


X_train, X_test, y_train, y_test = train_test_split(X_vars, y_var, test_size = 0.33)

LinReg = LinearRegression().fit(X_train, y_train)

y_pred = LinReg.predict(X_test)

r2_score(y_test, y_pred)



ols = sm.OLS(y_train, X_train)

ols_result = ols.fit()

ols_result.summary()


y_pred = ols_result.predict(X_test)

test_rmse = mean_squared_error(y_test, y_pred, squared = False)


#do an F test to see if multiple coefficients are significant













































































