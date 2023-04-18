import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, RocCurveDisplay, roc_curve
from sklearn.model_selection import KFold
import time
import matplotlib.pyplot as plt


MAX_VALUE = 2
USE_RANDOM_FOREST = False
SHOW_CURVE = False

def perc_format(num):
        return '{:.2f}%'.format(round(num * 100, 2))

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

    dr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    fnr = fn/(fn+tp)
    acc = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    print(f"DR: {dr} FPR: {fpr} FNR: {fnr} Acc: {perc_format(acc)} F1 Score: {f1}")
    print(f"FP: {fp} TP: {tp} FN: {fn} TN: {tn}")
    auc = np.round(roc_auc_score(y_test, y_pred), 3)
    print("Auc is {}". format(auc))
    return dr, fpr, fnr, f1


def ibgsa(X, y, n_agents, max_iter, G=6.6743, eps=0.01, k1 = 3, k2 = 13):
    # Initialization
    n_features = X.shape[1]
    agents = np.random.randint(2, size=(n_agents, n_features))
    mass = np.ones(n_agents)
    fitness = np.zeros(n_agents)
    best_agent = np.zeros(n_features, dtype=int)
    best_fitness = 0.0
    worst_fitness = 1.0
    velocity = np.zeros(n_features)
    failures = 0
    prev_best_agent = np.zeros(n_features, dtype=int)

    best_agents = np.ones(n_agents)
    modified_fitness = np.ones(n_agents)

    # Main loop
    for t in range(max_iter):
        currentGravity = G*(1-t/max_iter)
        for i in range(n_agents):
            # Evaluate fitness of current agent
            agent = agents[i]
            selected_features = X.iloc[:, agent == 1]

            X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.2, random_state=42)

            # Classifier change here
            if USE_RANDOM_FOREST:
                clf = RandomForestClassifier()
            else:
                clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            fitness[i] = accuracy_score(y_test, y_pred)

            # Update best agent
            if fitness[i] > best_fitness:
                prev_best_agent = best_agent.copy()
                best_agent = agents[i].copy()
                best_fitness = fitness[i]
                best_classifier = clf
                best_y_test = y_test
                best_y_pred = y_pred

            if fitness [i] < worst_fitness:
                worst_fitness = fitness[i]

        if t/(max_iter/n_agents) == 0:
            # Sort agents by fitness
            modified_fitness = [fitness[i] if best_agents[i] == 1 else MAX_VALUE for i in range(len(fitness))]   
            worst_agent_index = np.argmin(modified_fitness)
            if best_agents[worst_agent_index] != 0:
                best_agents[worst_agent_index] = 0
            else:
                print("Something went wrong")

        # Update mass of each agent
        mk = (fitness-worst_fitness)/(best_fitness-worst_fitness)
        mass = mk/np.sum(mk)

        if np.array_equal(prev_best_agent, best_agent):
            failures += 1
        else:
            failures = 0

        # Calculate force on each agent
        force = np.zeros((n_agents, n_features))
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j or best_agents[j] == 0:
                    continue
                # normalize hamming distance by dividing number of features
                distance = np.count_nonzero(agents[i]!=agents[j]) / n_features
                force[i] += np.random.rand() * currentGravity * mass[i] * mass[j] / (distance + eps) * (agents[j] - agents[i])

        # Update position of each agent
        for i in range(n_agents):
            if mass[i] == 0:
                continue
            acceleration = force[i] / mass[i]
            velocity = np.random.rand() * velocity + acceleration
            velocity[velocity > 6] = 6

            A = k1 * (1 - np.exp(failures / k2))

            ftemp = A + (1-A) * np.absolute(np.tanh(velocity))

            new_position = agents[i].copy()

            for pos in range(len(agents[i])):
                if np.random.rand() < ftemp[pos]:
                    new_position[pos] = 0 if agents[i][pos] == 1 else 1


            selected_features = X.iloc[:, new_position == 1]

            X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.2, random_state=42)

            # Classifier change here
            if USE_RANDOM_FOREST:
                clf = RandomForestClassifier()
            else:
                clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            new_fitness = accuracy_score(y_test, y_pred)

            if fitness[i] < new_fitness:
                agents[i] = new_position

    return best_agent, best_classifier, best_y_pred, best_y_test, best_fitness

program_start_time = time.time()

# Load data from CSV file
df = pd.read_csv('PDFMalware2022v2.csv')
le = LabelEncoder()
df = df.apply(lambda col: le.fit_transform(col) if col.dtype == object else col)

kf = KFold(n_splits=10, shuffle = True)
scores = []
drs = []
fprs = []
fnrs = []
f1s = []

# Separate features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

max_fitness = 0

fold_number = 1
tprs = []
base_fpr = np.linspace(0, 1, 101)

for train_index, test_index in kf.split(X):

    fold_start_time = time.time()

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Run IBGSA to select relevant features
    selected_features, selected_classifier, best_y_pred, best_y_test, best_fitness = ibgsa(X_train, y_train, n_agents=50, max_iter=100)

    # Print selected features
    print("Selected Features:")
    feature_columns = df.columns[:-1].tolist()
    selected_columns = [feature_columns[i] for i in range(len(selected_features)) if selected_features[i] == 1]
    print(selected_columns)

    # Train and test model using selected features
    selected_X_test = X_test.iloc[:, selected_features == 1]
    y_pred = selected_classifier.predict(selected_X_test)
    y_pred_prob = selected_classifier.predict_proba(selected_X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Output report about fold
    print("Fold Report:")
    dr, fprFromReport, fnr, f1 = report(y_test, y_pred)

    # ax = plt.gca()
    # rfc_disp = RocCurveDisplay.from_estimator(selected_classifier, selected_X_test, y_test, ax=ax, alpha=0.8)
    pred_y_score = selected_classifier.predict_proba(selected_X_test)
    fpr, tpr, _ = roc_curve(y_test, pred_y_score[:, 1])
    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)
    

    scores.append(accuracy)
    drs.append(dr)
    fnrs.append(fnr)
    fprs.append(fprFromReport)
    f1s.append(f1)

    if best_fitness > max_fitness:
        max_fitness = best_fitness
        max_y_test = best_y_test
        max_y_pred = best_y_pred
        max_features = selected_features
        max_classifier = selected_classifier

    print(f"--- Fold {fold_number} took {time.time() - fold_start_time} seconds --- " )
    fold_number += 1

print("=====FINAL RESULTS=====")

print("Best Report on fitness")
report(max_y_test, max_y_pred)

print(f"Using classifier {'Random forest' if USE_RANDOM_FOREST else 'Decision Tree'}" )
print("Average Results:")
print(f"DR: {np.mean(drs)} FPR: {np.mean(fprs)} FNR: {np.mean(fnrs)} F1 Score: {np.mean(f1s)} Acc: {perc_format(np.mean(scores))}")

print("!!! %s seconds in total !!!" % (time.time() - program_start_time))

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
plt.plot(base_fpr, mean_tprs)
if SHOW_CURVE:
    plt.show()

