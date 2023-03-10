import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def ibgsa(csv_path, iterations, G0, beta, k1, k2, classifier):
    # Load data from CSV file
    data = pd.read_csv(csv_path)

    # Encode categorical variables as integers
    le = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == object:
            data[column] = le.fit_transform(data[column])

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Split data into training and test sets
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    # Initialize population
    agents = 20
    n = X.shape[1]
    r = np.random.rand(agents, n)
    r_bin = np.round(r)
    fitness_vals = np.zeros(agents)
    best_fitness = -np.inf
    best_solution = np.zeros(n)
    prev_r_bin = np.zeros((agents, n))   # Store previous position for each candidate solution
    FC = 0                               # Initialize failure counter

    def fitness(solution):
        # Extract selected features
        selected_features = solution == 1
        if not any(selected_features):  # If no features are selected
            return 0, [], []

        X_train_selected = X_train[:, solution == 1]
        X_test_selected = X_test[:, solution == 1]

        # Train classifier depending on the type selected in IBGSA call
        if classifier == 'dt':
            clf = DecisionTreeClassifier(random_state=0)
        elif classifier == 'rf':
            clf = RandomForestClassifier(n_estimators=10, random_state=0)
        else:
            raise ValueError('Invalid classifier: {}'.format(classifier))

        clf.fit(X_train_selected, y_train)

        # Evaluate classifier accuracy on test set
        y_pred = clf.predict(X_test_selected)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
        fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
        return (tp + tn) / (tp + tn + fp + fn), fp_indices, fn_indices

    # Main loop
    for t in range(iterations):
        # Calculate fitness of each solution
        for i in range(agents):
            fitness_vals[i], false_positives, false_negatives = fitness(r_bin[i, :])
            if fitness_vals[i] > best_fitness:
                best_fitness = fitness_vals[i]
                best_solution = r_bin[i, :]
                best_false_positives = false_positives
                best_false_negatives = false_negatives

        # Update gravity
        G = G0 * (1 - (t/iterations))

        # Calculate gravitational forces
        F = np.zeros((agents, n))
        for i in range(agents):
            for j in range(n):
                for k in range(agents):
                    if k != i:
                        d = np.sum((r_bin[i, j] - r_bin[k, j]) ** 2)
                        F[i, j] += (r_bin[k, j] - r_bin[i, j]) * fitness_vals[k] / (d + 1e-10)

        # Update positions
        r_bin_new = np.zeros((agents, n))
        for i in range(agents):
            for j in range(n):
                is_elite = fitness_vals[i]
                r_bin_new[i, j] = np.round(r_bin[i, j] + is_elite * np.exp(-beta * G) * F[i, j])
                if r_bin_new[i, j] > 1:
                    r_bin_new[i, j] = 1
                elif r_bin_new[i, j] < 0:
                    r_bin_new[i, j] = 0

        # Update failure counter
        FC = FC + 1 if np.array_equal(r_bin_new[i, :], prev_r_bin[i, :]) else 0
        prev_r_bin[i, :] = r_bin[i, :]

        # Calculate A based on FC
        A = k1 * (1 - np.exp(-FC/k2))

        # Apply stagnation avoidance
        r_bin_new = A + (1 - A) * np.tanh(r_bin_new)

        # Update population
        r_bin = r_bin_new.copy()

    # Get indices of selected features in best solution
    selected_indices = np.where(best_solution == 1)[0]

    # Get names of selected features
    feature_names = data.columns[:-1][selected_indices].tolist()

    # Print results
    print("===========")
    print("Classifier: ", classifier)
    print("Amount of features: ", len(feature_names))
    print("Selected features: ", feature_names)
    print("Best fitness: ", best_fitness)
    print("# False positives: ", len(best_false_positives))
    print("# False negatives: ", len(best_false_negatives))

    return best_solution


start_time = time.time()
ibgsa_result = ibgsa('PDFMalware2022v2.csv', iterations=100, G0=10, beta=10, k1=0.1, k2=10, classifier='dt')
end_time = time.time()
print("Time taken: {:.2f} seconds".format(end_time - start_time))

print("===========")

start_time = time.time()
ibgsa_result = ibgsa('PDFMalware2022v2.csv', iterations=100, G0=10, beta=10, k1=0.1, k2=10, classifier='rf')
end_time = time.time()
print("Time taken: {:.2f} seconds".format(end_time - start_time))
