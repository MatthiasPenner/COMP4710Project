import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier

# Load the CSV file into a Pandas DataFrame
data = pd.read_csv('../Public/PDFMalware2022Grepped.csv')

# Apply the LabelEncoder
le = LabelEncoder()
for column in data.columns:
	if data[column].dtype == object:
		data[column] = le.fit_transform(data[column])

# Split the data into features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets with an 80:20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr = LogisticRegression(solver='newton-cg', max_iter=20000)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
ext = ExtraTreesClassifier(random_state=42)
ada = AdaBoostClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)


classifiers = [[gb, 'Gradient Boosting'], [ada, 'Ada Boost'], [ext, 'Extra Trees'], [lr, 'Logistic Regression'], [dt, 'Decision Trees'], [rf, 'Random Forest']]


# No feature selection algorithm
for classifier in classifiers:

	accuraciesLocal = [0]*40


	for j in range(0, 5):

		classifier[0].fit(X_train, y_train)

		y_pred = classifier[0].predict(X_test)

		accuracy = accuracy_score(y_test, y_pred)

		accuraciesLocal[j] = accuracy


	accuracy=np.max(np.array(accuraciesLocal))

	print('Max accuracy for '+classifier[1]+' with no selection algorithm: '+ str(np.max(np.array(accuracy))))



# XGBoost Feature Selection
for classifier in classifiers:

	accuracies = [0]*33

	for i in range(5, X_train.shape[1]+1):

		accuraciesLocal = [0]*40

		for j in range(0, 10):

			xgb = XGBClassifier()
			
			xgb.fit(X_train, y_train)

			feature_importances = xgb.feature_importances_

			sorted_idx = np.argsort(feature_importances)[::-1]

			selected_features = sorted_idx[:i].tolist()
            
			X_train_selected = X_train.iloc[:, selected_features]

			X_test_selected = X_test.iloc[:, selected_features]

			classifier[0].fit(X_train_selected, y_train)

			y_pred = classifier[0].predict(X_test_selected)

			accuracy = accuracy_score(y_test, y_pred)

			accuraciesLocal[j] = accuracy

		accuracies[i] = np.max(np.array(accuraciesLocal))

		print('Max accuracy for i='+str(i)+': '+str(accuracies[i]))

	print('Max accuracy for '+classifier[1]+' with XGBoost feature selection: '+ str(np.max(np.array(accuracies))))


# Recursive Feature Elimination
for classifier in classifiers:
	accuracies = [0]*33
	for i in range(5, 33):

		accuraciesLocal = [0]*40

		for j in range(0, 5):
			rfe = RFE(classifier[0], n_features_to_select=i)

			rfe.fit(X_train, y_train)

			y_pred = rfe.predict(X_test)

			accuracy = accuracy_score(y_test, y_pred)

			accuraciesLocal[j] = accuracy

		accuracies[i]=np.max(np.array(accuraciesLocal))

		print('Max accuracy for i='+str(i)+': '+str(accuracies[i]))

	print('Max accuracy for '+classifier[1]+' with RFE: '+ str(np.max(np.array(accuracies))))


# ANOVA F-Test
for classifier in classifiers:
	accuracies = [0]*33
	for i in range(5, 33):

		accuraciesLocal = [0]*40

		for j in range(0, 5):
			selector = SelectKBest(f_classif, k=i)

			X_train_selected = selector.fit_transform(X_train, y_train)

			X_test_selected = selector.transform(X_test)

			classifier[0].fit(X_train_selected, y_train)

			y_pred = classifier[0].predict(X_test_selected)

			accuracy = accuracy_score(y_test, y_pred)

			accuraciesLocal[j] = accuracy

		accuracies[i]=np.max(np.array(accuraciesLocal))

		print('Max accuracy for i='+str(i)+': '+str(accuracies[i]))

	print('Max accuracy for '+classifier[1]+' with ANOVA F-Test: '+ str(np.max(np.array(accuracies))))


# Pearson Correlation
for classifier in classifiers:
	accuracies = [0]*40
	for i in range(5, 33):
		accuraciesLocal = [0]*10
		
		X = data.drop('Class', axis=1)
		
		# Calculate Pearson Correlation between each feature and the target variable
		correlations = X.corrwith(y)

		n_features = i
		top_features = abs(correlations).sort_values(ascending=False)[:n_features].index

		X = X[top_features]

		# Split the data into training and testing sets
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
		for j in range(0, 10):
			
            # Use selected features for training and testing
			sel = SelectFromModel(classifier[0])
			sel.fit(X_train, y_train)
			X_train_new = sel.transform(X_train)
			X_test_new = sel.transform(X_test)

			classifier[0].fit(X_train_new, y_train)
			y_pred = classifier[0].predict(X_test_new)
			accuracy = accuracy_score(y_test, y_pred)
			accuraciesLocal[j] = accuracy

		accuracies[i]=np.max(np.array(accuraciesLocal))
		print('Max accuracy for i='+str(i)+': '+str(accuracies[i]))
		
	print('Max accuracy for '+classifier[1]+' with Pearson Corellation: '+ str(np.max(np.array(accuracies))))
