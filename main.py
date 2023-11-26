import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from DecisionTree import DecisionTreeRegressor, DecisionTreeClassifier
from RandomForest import RandomForestRegressor


# data = pd.read_csv("./data/IRIS.csv")

# X = data.drop(columns=["species"])
# y = data.species

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# model = DecisionTreeClassifier()

# model.fit(X_train, y_train)

# predictions = model.predict(X_val)

# count = 0
# for i, ans in enumerate(y_val):
#     print(ans + "\t" + predictions[i])
#     if ans == predictions[i]:
#         count+=1

# print(count)


data = pd.read_csv("./data/AirfoilSelfNoise.csv")

X = data.drop(columns=["SSPL"])
y = data.SSPL

model = RandomForestRegressor()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)

predictions = model.predict(X_val)

mean_absolute_error = 0
for i, ans in enumerate(y_val):
    mean_absolute_error += abs(ans-predictions[i])
    print(ans, predictions[i], ans-predictions[i])

print(mean_absolute_error/X_val.shape[0])

# 2.190040041132733 for 3 trees
# 1.777299028041447 for 7 trees
# 1.896338682170542 for 10 trees
