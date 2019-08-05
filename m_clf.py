import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

plt.style.use('seaborn-notebook')
print(plt.style.available)

df = pd.read_csv('Dataset/mushrooms.csv')
print(df.head())

df = df[['class', 'odor']]
print(df.head())

odor_types = df['odor'].unique()
odor_count = np.array(df['odor'].value_counts())
print(odor_types)
print(odor_count)
  
print(df.head())
X = df['odor']
y = df['class']

X = LabelEncoder().fit_transform(X).reshape(-1, 1)
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lregressor = LogisticRegression()
lregressor.fit(X_train, y_train)

y_pred = lregressor.predict(X_test)
results = pd.DataFrame({'odor': X_test.flatten(),
                        'class(1=p, e=0)': y_pred.flatten()})
results.to_csv('Dataset/results.csv')

print("accuracy: " +  str(lregressor.score(X, y)))
print("Cost: " + str(metrics.mean_squared_error(X, y)))

#print(results.head())
#print(results['odor'].unique())
#print(results['class(1=p, e=0)'].value_counts())

plt.figure(1)
plt.bar(['edible','poisonous'], np.array(results['class(1=p, e=0)'].value_counts()))
plt.ylabel('count')

plt.figure(2)
plt.scatter(X_test, y_test)
plt.xlabel('types of smells')
plt.ylabel('poisonous?')
plt.show()

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
