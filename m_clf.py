import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import sklearn.model_selection

plt.style.use('seaborn-notebook')

# viewing the data
df = pd.read_csv('Dataset/mushrooms.csv')
print(df.head())
print(df['gill-color'].value_counts())
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


# visualising
print(results['class'].value_counts())
plt.figure(1)
plt.bar(['Edible','Poisonous'], np.array(results['class'].value_counts()))
plt.ylabel('Count')
plt.title('Predicted counts of poisonous mushooms and edible mushrooms')
plt.show()

plt.figure(2)
results = pd.DataFrame(results[['class', 'gill-color', 'cap-color']])
groupby_class = results.groupby(['class'])

for name, group in groupby_class:
    plt.plot(group['gill-color'], group['cap-color'],marker='o', linestyle='', markersize='12', label=name)
plt.xlabel('gill-color')
plt.ylabel('cap-color')
plt.legend()
plt.show()
