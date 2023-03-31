import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def ibgsa(csv_path, iterations, G0, alpha, beta, classifier):
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
    mass = 20
    n = X.shape[1]  # Total # Features
    r = np.random.rand(mass, n)  # Random Number between mass + n features
    r_bin = np.round(r)
    fitness_vals = np.zeros(mass)
    best_fitness = -np.inf
    best_solution = np.zeros(n)

    def perc_format(num):
        return '{:.2f}%'.format(round(num * 100, 2))

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

        return (tp + tn) / (tp + tn + fp + fn), fp, fn, tn, tp

    # Main loop
    for t in range(iterations):
        # Calculate fitness of each solution
        for i in range(mass):
            fitness_vals[i], fp, fn, tn, tp = fitness(r_bin[i, :])
            if fitness_vals[i] > best_fitness:
                best_fitness = fitness_vals[i]
                best_solution = r_bin[i, :]
                best_false_positives = fp
                best_false_negatives = fn
                best_true_positives = tp
                best_true_negatives = tn

        # Update gravity
        G = G0 * (1 - (t / iterations))

        # Update positions and calculate forces
        r_bin_new = np.zeros((mass, n))
        F = np.zeros((mass, n))
        for i in range(mass):
            for j in range(n):
                is_elite = fitness_vals[i]
                for k in range(mass):
                    if k != i:
                        d = np.sum((r_bin[i, j] - r_bin[k, j]) ** 2)
                        d_norm = np.sum(r_bin[i, :] != r_bin[k, :]) / max(np.sum(r_bin[i, :]), np.sum(r_bin[k, :]))
                        F[i, j] += (r_bin[k, j] - r_bin[i, j]) * fitness_vals[k] / (
                                    (d_norm + 1e-10) * (d + 1e-10) ** 0.5)
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

    precision = best_true_positives / (best_true_positives + best_false_positives)
    recall = best_true_positives / (best_true_positives + best_false_negatives)
    f1 = 2 * precision * recall / (precision + recall)
    detect_rate = best_true_positives / (best_true_positives + best_false_negatives)

    # Print results
    print("===========")
    print("Classifier: ", classifier)
    print("Amount of features: ", len(feature_names))
    print("Selected features: ", feature_names)
    print("Best fitness: ", perc_format(best_fitness))
    print("# False positives: ", best_false_positives)
    print("# False negatives: ", best_false_negatives)
    print("# True positives: ", best_true_positives)
    print("# True negatives: ", best_true_negatives)
    print("F1 Score: ", perc_format(f1))
    print("Detection Rate: ", perc_format(detect_rate))

    return best_solution


start_time = time.time()
ibgsa_result = ibgsa('PDFMalware2022v2.csv', iterations=100, G0=10, alpha=0.1, beta=10, classifier='dt')
end_time = time.time()
print("Time taken: {:.2f} seconds".format(end_time - start_time))

print("===========")

start_time = time.time()
ibgsa_result = ibgsa('PDFMalware2022v2.csv', 100, 10, 0.1, 10, 'rf')
end_time = time.time()
print("Time taken: {:.2f} seconds".format(end_time - start_time))
