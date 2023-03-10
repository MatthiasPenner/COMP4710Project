import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def ibgsa(csv_path, iterations, G0, Gf, alpha, beta, classifier):
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
    m = 20
    n = X.shape[1]
    r = np.random.rand(m, n)
    r_bin = np.round(r)
    fitness_vals = np.zeros(m)
    best_fitness = -np.inf
    best_solution = np.zeros(n)

    def fitness(solution):
        # Extract selected features
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
        for i in range(m):
            fitness_vals[i], false_positives, false_negatives = fitness(r_bin[i, :])
            if fitness_vals[i] > best_fitness:
                best_fitness = fitness_vals[i]
                best_solution = r_bin[i, :]
                best_false_positives = false_positives
                best_false_negatives = false_negatives

        # Update gravity
        G = G0 - t * (G0 - Gf) / iterations

        # Calculate gravitational forces
        F = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                for k in range(m):
                    if k != i:
                        d = np.sum((r_bin[i, j] - r_bin[k, j]) ** 2)
                        F[i, j] += (r_bin[k, j] - r_bin[i, j]) * fitness_vals[k] / (d + 1e-10)

        # Update positions
        r_bin_new = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                is_elite = fitness_vals[i]
                r_bin_new[i, j] = np.round(r_bin[i, j] + is_elite * alpha * np.exp(-beta * G) * F[i, j])
                if r_bin_new[i, j] > 1:
                    r_bin_new[i, j] = 1
                elif r_bin_new[i, j] < 0:
                    r_bin_new[i, j] = 0

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
ibgsa_result = ibgsa('PDFMalware2022v2.csv', iterations=100, G0=10, Gf=0.1, alpha=0.1, beta=10, classifier='dt')
end_time = time.time()
print("Time taken: {:.2f} seconds".format(end_time - start_time))

print("===========")

start_time = time.time()
ibgsa_result = ibgsa('PDFMalware2022v2.csv', 100, 10, 0.1, 0.1, 10, 'rf')
end_time = time.time()
print("Time taken: {:.2f} seconds".format(end_time - start_time))
