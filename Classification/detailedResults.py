import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
import time
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics


startTime = time.time()

data = pd.read_csv('../Public/PDFMalware2022Grepped.csv')

le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == object:
        data[column] = le.fit_transform(data[column])


X = data.drop('Class', axis=1)
y = data['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
ext = ExtraTreesClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

def report(y_test, y_pred):
    tp, tn, fp, fn = 0, 0, 0, 0
    for x in range(len(y_test)):
        actual = list(y_test)[x]
        predicted  = list(y_pred)[x]
        if actual == 1 and predicted == 1:
            tp+=1
        elif actual == 0 and predicted == 0:
            tn+=1
        elif actual == 1 and predicted == 0:
            fn += 1
        else:
            fp+=1

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    f1 = (2*precision*recall)/(precision+recall)
    dr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    fnr = fn/(fn+tp)
    acc = (tp+tn)/(tp+tn+fp+fn)
    print(f"DR: {dr} FPR: {fpr} FNR: {fnr} Acc: {acc} F1: {f1}")


startTime = time.time()

rfe = RFE(gb, n_features_to_select=13)

rfe.fit(X_train, y_train)

y_pred = rfe.predict(X_test)


print("Time: "+str(time.time()-startTime))
print("Report for Gradient Boosting with RFE (13 features)")
report(y_test, y_pred)
ax = plt.gca()

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
rfc_disp = RocCurveDisplay.from_estimator(rfe, X_test, y_test, ax=ax, alpha=0.8, name='_')
rfc_disp.line_.remove()
rfc_disp.estimator_name = 'Gradient Boosting + RFE AUC={:.4f}'.format(roc_auc)
rfc_disp.roc_auc = None
rfc_disp.plot(ax=ax, alpha=0.8, color='green')


startTime = time.time()

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
feature_importances = xgb.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]
selected_features = sorted_idx[:17].tolist()
X_train_selected = X_train.iloc[:, selected_features]
X_test_selected = X_test.iloc[:, selected_features]
ext.fit(X_train_selected, y_train)
y_pred = ext.predict(X_test_selected)


print("Time: "+str(time.time()-startTime))
print("Report for Extra Trees with XGBoost feature selection (17 features)")


report(y_test, y_pred)

ax = plt.gca()
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
rfc_disp = RocCurveDisplay.from_estimator(ext, X_test_selected, y_test, ax=ax, alpha=0.8, name='_')
rfc_disp.line_.remove()
rfc_disp.estimator_name = 'Extra Trees with XGBoost feature selection AUC={:.4f}'.format(roc_auc)
rfc_disp.roc_auc = None
rfc_disp.plot(ax=ax, alpha=0.8, color='red')

startTime = time.time()

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
feature_importances = xgb.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]
selected_features = sorted_idx[:17].tolist()
X_train_selected = X_train.iloc[:, selected_features]
X_test_selected = X_test.iloc[:, selected_features]
rf.fit(X_train_selected, y_train)
y_pred = rf.predict(X_test_selected)


print("Time: "+str(time.time()-startTime))
print("Report for Random Forest with XGBoost feature selection (17 features)")


report(y_test, y_pred)

ax = plt.gca()

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
rfc_disp = RocCurveDisplay.from_estimator(rf, X_test_selected, y_test, ax=ax, alpha=0.8, name='_')
rfc_disp.line_.remove()
rfc_disp.estimator_name = 'Random Forest with XGBoost feature selection AUC={:.4f}'.format(roc_auc)
rfc_disp.roc_auc = None
rfc_disp.plot(ax=ax, alpha=0.8, color='orange')


startTime = time.time()
rfe = RFE(rf, n_features_to_select=29)

rfe.fit(X_train, y_train)

y_pred = rfe.predict(X_test)


print("Time for Random Forest with RFE: "+str(time.time()-startTime))
print("Report for Random Forest with RFE (29 features)")

report(y_test, y_pred)

ax = plt.gca()

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
rfc_disp = RocCurveDisplay.from_estimator(rfe, X_test, y_test, ax=ax, alpha=0.8, name='_')
rfc_disp.line_.remove()
rfc_disp.estimator_name = 'Random Forest with RFE AUC={:.4f}'.format(roc_auc)
rfc_disp.roc_auc = None
rfc_disp.plot(ax=ax, alpha=0.8, color='blue')


startTime = time.time()
rfe = RFE(ext, n_features_to_select=16)

rfe.fit(X_train, y_train)

y_pred = rfe.predict(X_test)

print("Time: "+str(time.time()-startTime))
print("Report for Extra Trees with RFE (16 features)")

report(y_test, y_pred)

ax = plt.gca()

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
rfc_disp = RocCurveDisplay.from_estimator(rfe, X_test, y_test, ax=ax, alpha=0.8, name='_')
rfc_disp.line_.remove()
rfc_disp.estimator_name = 'Extra Trees with RFE AUC={:.4f}'.format(roc_auc)
rfc_disp.roc_auc = None
rfc_disp.plot(ax=ax, alpha=0.8, color='green')

plt.xlim([0, 0.055])
plt.ylim([0.97, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic Curve")
plt.show()