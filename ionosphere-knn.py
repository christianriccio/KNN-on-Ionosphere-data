import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

reader= pd.read_excel('data.xlsx')


X= reader.iloc[:, 0: -1]
y= reader.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state= 14)

model= KNeighborsClassifier()
model.fit(X_train, y_train)

y_pred= model.predict(X_test)
accuracy=np.mean(y_test==y_pred)*100
print(f"Accuracy: {np.round(accuracy, 4)}")
accuracy=accuracy_score(y_test, y_pred)
print("")
print(f"Accuracy with sklearn: {np.round(accuracy, 4)}")

# introducing CV
scores= cross_val_score(model, X, y, scoring='accuracy')

score= np.mean(scores)
print(f"With the K-fold CV the average score is: {np.round(score, 4)}")

# set the idelas number of n_neighbors 

avg_scores=[]
all_scores=[]

values= list(range(1,21))
print()
for value in values:
    model=KNeighborsClassifier(n_neighbors=value)
    scores=cross_val_score(model, X, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)

print(f"Avg scores: {avg_scores}")
print(f"All scores: {all_scores}")

# plot the relationship between number of neighbors and accuracy
plt.plot(values, avg_scores, '-o')
plt.show()

# pre-processing
X_change=np.array(X)

X_change[:, ::2]/=10

model=KNeighborsClassifier()
original_scores=cross_val_score(model, X,y, scoring='accuracy')
print(f"The original average accuracy is {np.mean(original_scores)*100}")
change_scores=cross_val_score(model, X_change, y, scoring='accuracy')
print()
print(f"The modified average accuracy is {np.mean(change_scores)*100}")

# so the tecnique of dicide by tean only the even fetures leads to a decrease of accuracy
# I perform a features based normalization, eache feature is scaled between 0 and 1
# The smallest value is replaced with 0 the largest with 1 and in the range lives all the other values

X_transformed= MinMaxScaler().fit_transform(X)
model=KNeighborsClassifier()
transformed_scores=cross_val_score(model, X_transformed, y, scoring='accuracy')
print(f"The average accuracy score is: {np.mean(transformed_scores)*100}")

# implement the Pipeline

scaling_pipeline= Pipeline([('scale', MinMaxScaler()), ('predict', KNeighborsClassifier())])
scores= cross_val_score(scaling_pipeline, X, y, scoring='accuracy')
print(f"From the pipeline the accuracy is: {np.mean(scores)*100}")



