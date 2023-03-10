import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

officialIBGSA = False

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
    acc = (tp+tn)/(tp+tn+fp+fn)
    print(f"DR: {dr} FPR: {fpr} Acc: {acc}")

def ibgsa(X, y, n_agents, max_iter, G=6.6743e-11, eps=0.01, k1 = 3, k2 = 13):
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

    # Main loop
    for t in range(max_iter):
        currentGravity = G*(1-t/max_iter)
        print("Now processing iter: " + str(t))
        for i in range(n_agents):
            # Evaluate fitness of current agent
            agent = agents[i]
            selected_features = X.iloc[:, agent == 1]

            X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.3, random_state=42)
            clf = DecisionTreeClassifier()

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            fitness[i] = accuracy_score(y_test, y_pred)

            # Update best agent
            if fitness[i] > best_fitness:
                prev_best_agent = best_agent.copy()
                best_agent = agents[i].copy()
                best_fitness = fitness[i]

            if fitness [i] < worst_fitness:
                worst_fitness = fitness[i]

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
                if i == j:
                    continue
                # normalize hamming distance by dividing number of features
                distance = np.count_nonzero(agents[i]!=agents[j]) / n_features
                # print(f"Distance computed is {distance1} vs {distance}")
                force[i] += np.random.rand() * currentGravity * mass[i] * mass[j] / (distance + eps) * (agents[j] - agents[i])
            # print(f"total force on agent {i} is {np.sum(force[i])}")
        # Update position of each agent
        for i in range(n_agents):
            if mass[i] == 0:
                print(f"No mass for agent {i}")
                continue
            # print(f"Agent {i}: force: {force[i]} mass: {mass[i]}")
            acceleration = force[i] / mass[i]

            if officialIBGSA: #currently broken
                velocity = np.random.rand() * velocity + acceleration
                # print(f"Velocity of agent {i} before limit is {velocity}")
                velocity[velocity > 6] = 6
                # print(f"Velocity of agent {i} after limit is {velocity}")

                A = k1 * (1 - np.exp(failures / k2))

                ftemp = A + (1-A) * np.absolute(np.tanh(velocity))

                new_position = agents[i].copy()

                for pos in range(len(agents[i])):
                    if np.random.rand() < ftemp[pos]:
                        # print(f"Flipping agent {i} ")
                        new_position[pos] = 0 if agents[pos] == 1 else 1

            else:
                velocity = np.random.uniform(low=-eps, high=eps, size=n_features) + acceleration
                new_position = np.logical_xor(agents[i], np.random.binomial(1, np.abs(velocity))).astype(int)


            selected_features = X.iloc[:, new_position == 1]

            X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.3, random_state=42)
            clf = DecisionTreeClassifier()

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            new_fitness = accuracy_score(y_test, y_pred)

            if fitness[i] < new_fitness:
                agents[i] = new_position

        print(f"Fitness value: {best_fitness} from agent {best_agent} ")
    return best_agent

# Load data from CSV file
df = pd.read_csv('PDFMalware2022v2.csv')
le = LabelEncoder()
df = df.apply(lambda col: le.fit_transform(col) if col.dtype == object else col)

# Separate features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Run IBGSA to select relevant features
selected_features = ibgsa(X, y, n_agents=50, max_iter=100)

# Print selected features
print("Selected Features:")
print(selected_features)

# Train and test model using selected features
selected_X = X.iloc[:, selected_features == 1]
X_train, X_test, y_train, y_test = train_test_split(selected_X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
report(y_test, y_pred)