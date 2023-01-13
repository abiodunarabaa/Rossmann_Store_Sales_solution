from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.optimizers import gradient_descent_v2



train = pd.read_csv('C:/rossmann-store-sales/train.csv', low_memory = False)
store = pd.read_csv('C:/rossmann-store-sales/store.csv', low_memory = False)
test = pd.read_csv('C:/rossmann-store-sales/test.csv', low_memory = False)

print ("train dataset by rows and columns: ", train.shape)
print ("Store dataset by rows and columns: ", store.shape)
print ("test dataset by rows and columns: ", test.shape)

train.head(2)
store.head(2)
test.head(2)

#Datetime conversion in train data set
dt_corrected_train = pd.to_datetime(train['Date'], format="%Y-%m-%d", errors='raise')


#Datetime conversion in test data set
dt_corrected_test = pd.to_datetime(test['Date'], format="%Y-%m-%d", errors='raise')

test['Date'] = dt_corrected_test

train['Date']= dt_corrected_train

train.info()

train["Year"] = train["Date"].dt.year
train["Month"] = train["Date"].dt.month
train["DayOfMonth"] = train["Date"].dt.day

train.describe()
test.info()

store["CompetitionDistance"].fillna(store["CompetitionDistance"].median(), inplace = True)
store.fillna(0, inplace = True)


merged_test = test.merge(store, how='left', on="Store", validate="many_to_one")

merged_train = train.merge(store, how='left', on="Store", validate="many_to_one")

merged_train.info()
merged_train.describe()


print("Rows before dropping duplicates: " + str(merged_train.shape[0]))
merged_train = merged_train.drop_duplicates()
print("Rows after dropping duplicates: " + str(merged_train.shape[0]))

print("Rows before dropping duplicates: " + str(merged_test.shape[0]))
merged_test = merged_test.drop_duplicates()
print("Rows after dropping duplicates: " + str(merged_test.shape[0]))



#Using str.strip() to remove white spaces.
for col in merged_train:
    if merged_train[col].dtype == object:
        merged_train[col] = merged_train[col].str.strip()

for col in merged_train:
    if merged_train[col].dtype == object:
        print(merged_train[col].value_counts())

merged_train.skew()


merged_train.plot (y= ["Sales", "Customers", "Open", "SchoolHoliday","CompetitionOpenSinceYear",  "CompetitionDistance", "CompetitionOpenSinceMonth" ], 
                   layout = (4,2), subplots =  True, kind= "box", figsize=(15,15) )

merged_train['CompetitionOpenSinceMonth'] = merged_train['CompetitionOpenSinceMonth'].convert_dtypes()
merged_train['CompetitionOpenSinceyear'] = merged_train['CompetitionOpenSinceYear'].convert_dtypes()

merged_train['Promo2SinceWeek'] = merged_train['Promo2SinceWeek'].convert_dtypes()
merged_train['Promo2SinceYear'] = merged_train['Promo2SinceYear'].convert_dtypes()

merged_train['StoreType'].replace(['a','b','c','d'], [1,2,3,4], inplace=True)
merged_train['Assortment'].replace(['a','b','c'], [1,2,3], inplace=True)

merged_train['StateHoliday'].replace(['0','a','b','c'], [0,1,2,3], inplace=True)
merged_train['PromoInterval'].replace(['Jan,Apr,Jul,Oct','Feb,May,Aug,Nov','Mar,Jun,Sept,Dec'], [1,2,3], inplace=True)

merged_train.info()
corr = merged_train.corr()
plt.figure(figsize = (15,8))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, center=0, cmap="YlGnBu", annot=True)
plt.show()


merged_train.fillna(0, inplace = True)
merged_train.isnull().sum()

merged_train.info
merged_train


def calculate_outlier(df,column):
    Q3 = df[column].quantile(0.75)
    Q1 = df[column].quantile(0.25)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper


col = "Sales"
lower_ins,upper_ins = calculate_outlier(merged_train,col)
print(lower_ins,upper_ins)
merged_train[(merged_train[col]>upper_ins) | (merged_train[col]<lower_ins)].count()


merged_train.loc[(merged_train[col]>upper_ins) | (merged_train[col]<lower_ins),col] = upper_ins
merged_train[(merged_train[col]>upper_ins) | (merged_train[col]<lower_ins)].count()

col = "Customers"
lower_ins,upper_ins = calculate_outlier(merged_train,col)
print(lower_ins,upper_ins)
merged_train[(merged_train[col]>upper_ins) | (merged_train[col]<lower_ins)].count()

merged_train.loc[(merged_train[col]>upper_ins) | (merged_train[col]<lower_ins),col] = upper_ins
merged_train[(merged_train[col]>upper_ins) | (merged_train[col]<lower_ins)].count()

col = "CompetitionDistance"
lower_ins,upper_ins = calculate_outlier(merged_train,col)
print(lower_ins,upper_ins)
merged_train[(merged_train[col]>upper_ins) | (merged_train[col]<lower_ins)].count()

merged_train.loc[(merged_train[col]>upper_ins) | (merged_train[col]<lower_ins),col] = upper_ins
merged_train[(merged_train[col]>upper_ins) | (merged_train[col]<lower_ins)].count()

col = "CompetitionOpenSinceYear"
lower_ins,upper_ins = calculate_outlier(merged_train,col)
print(lower_ins,upper_ins)
merged_train[(merged_train[col]>upper_ins) | (merged_train[col]<lower_ins)].count()

merged_train.loc[(merged_train[col]>upper_ins) | (merged_train[col]<lower_ins),col] = upper_ins
merged_train[(merged_train[col]>upper_ins) | (merged_train[col]<lower_ins)].count()

merged_train.plot (y= ["Sales", "Customers", "Open", "SchoolHoliday","CompetitionOpenSinceYear",  "CompetitionDistance", "CompetitionOpenSinceMonth" ], 
                   layout = (4,2), subplots =  True, kind= "box", figsize=(15,15) )

merged_train_array = np.asarray(merged_train)
merged_train_array

mergedMin = np.min(merged_train_array[:,:], axis = 0)
mergedMax = np.max(merged_train_array[:,:], axis = 0)
mergedNorm = (merged_train_array - mergedMin) / (mergedMax - mergedMin)
mergedNorm

Y = mergedNorm[:,2]


DofWeek = mergedNorm[:,1]
cust = mergedNorm[:,3]
openSt = mergedNorm[:,4]
promo = mergedNorm[:,5]


X = np.column_stack((DofWeek, cust, openSt, promo))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train
X_test
Y_train
Y_test




# Reshaping X_train for efficient modelling
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))


X_train = np.asarray(X_train,dtype='float')
Y_train = np.asarray(Y_train,dtype='float')



X_test = np.asarray(X_test,dtype='float')
Y_test = np.asarray(Y_test,dtype='float')




# Libraries for this section


from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error


sc = MinMaxScaler(feature_range=(0,1))



# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
# Second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Fourth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='rmsprop',loss='mean_squared_error')
# Fitting to the training set
regressor.fit(X_train,Y_train,epochs=20,batch_size=32)



X_test = np.array(X_test)
#X_scaled = scaler.fit_transform(X_test)
#obj = scaler.fit(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_sales = regressor.predict(X_test)
print(predicted_sales)
#predicted_stock_price = obj.inverse_transform(predicted_sales)

prediction = pd.DataFrame(predicted_sales, columns=['Sales']).to_csv('submission.csv')

pred_sales_norm = np.asarray(predicted_sales) #the conversion of the dataset to array

Ymin = np.min(pred_sales_norm[:,:], axis= 0)
Ymax = np.max(pred_sales_norm[:,:], axis= 0)

pred_sales_denorm =(pred_sales_norm) * (Ymax - Ymin) + Ymin 

print (pred_sales_denorm)




