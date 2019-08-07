import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

plt.style.use('seaborn-notebook')

# viewing the data
df = pd.read_csv('Dataset/mushrooms.csv')
print(df.head())

encoder = LabelEncoder()

# X =  all cols apart from class
# y = class(represents poisonous or edible as p/e) 
X = df.drop(['class'], axis=1)
y = df['class']

# data is represented in letters. The encoder converts the letters into values
for col in X.columns:
    X[col] = encoder.fit_transform(X[col]) #encoder can only take vectors not matrices hence split the cols in a for loops

X = np.array(X)
y = encoder.fit_transform(y)
#print(X.shape)
#print(y.shape)

# split the data 80% we will use to train 20% is used to test the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train and predict 
lregressor = LogisticRegression()
lregressor.fit(X_train, y_train)
y_pred = lregressor.predict(X_test)

# print the cost and accuracy
print("accuracy: " +  str(lregressor.score(X, y)))
print("Cost: " + str(metrics.mean_squared_error(y_test, y_pred)))

# add the X_test vals in a dictionary
results_dict = {}
cols = df.columns.drop(['class'])
for i in range(np.size(df.columns)-1):
    results_dict[cols[i]] = X_test[:, [i]].flatten()

# upload the results to a csv
results_dict.update({'class': y_pred})
results = pd.DataFrame(results_dict)
results.to_csv('Dataset/results.csv')

#print(results.head())
#print(results['odor'].unique())
#print(results['class(1=p, e=0)'].value_counts())

#splt.figure(1)

'''
plt.figure(1)
plt.bar(['edible','poisonous'], np.array(results['class(1=p, e=0)'].value_counts()))
plt.ylabel('count')

plt.figure(2)
plt.scatter(X_test, y_test)
plt.xlabel('types of smells')
plt.ylabel('poisonous?')
plt.show()
'''
'''
plt.figure(1)
plt.bar(odor_types, odor_count, color='green')
plt.title('counts for each mushroom odor')
plt.xlabel('odor')
plt.ylabel('count')

plt.figure(2)
plt.bar(['edible', 'poisonous'], np.array(df['class'].value_counts()), color='red')
plt.title('counts for each class')
plt.xlabel('class')
plt.ylabel('count')

plt.figure(3)
plt.scatter(X, y)

plt.show()
'''
