import random


def initialize_weights(num_attributes):
    min_num = -1
    max_num = 1

    initialized_weights = [random.uniform(min_num, max_num) for _ in range(num_attributes)]
    initialized_threshold = random.uniform(min_num, max_num)

    return initialized_weights, initialized_threshold


def map_labels(actual_output):
    label_mapping_ = {}

    unique_labels = set(actual_output)

    for i, label in enumerate(unique_labels):
        label_mapping_[label] = i

    return label_mapping_


def read_file(file):
    input_features = []
    labels = []

    with open(file) as f:
        for line in f:
            parts = line.strip().split(',')
            line = [float(x) for x in parts[:-1]]
            labels.append(parts[-1])
            input_features.append(line)

    return input_features, labels


def calculate_weighted_sum(x, w):
    return sum(xi * wi for xi, wi in zip(x, w))


def activation_function(y, activation_threshold):
    if y >= activation_threshold:
        output = 1
    else:
        output = 0

    return output


def calculate_accuracy(predictions, labels):
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    total = len(predictions)
    accuracy = correct / total
    return accuracy


def train_perceptron(learning_rate, epochs, train_file_, test_file_):
    input_features, labels = read_file(train_file_)
    label_mapping_ = map_labels(labels)
    weights_, threshold_ = initialize_weights(len(input_features[0]))

    test_input_features, test_labels = read_file(test_file_)

    for epoch in range(epochs):
        combined_data = list(zip(input_features, labels))
        random.shuffle(combined_data)
        input_features_shuffled, labels_shuffled = zip(*combined_data)

        for features, label in zip(input_features_shuffled, labels_shuffled):
            actual_output = label_mapping_[label]
            y = calculate_weighted_sum(features, weights_)
            prediction = activation_function(y, threshold_)
            error = actual_output - prediction

            for i in range(len(weights_)):
                weights_[i] += learning_rate * error * features[i]

        test_predictions = [list(label_mapping_.keys())[list(label_mapping_.values()).index(
            activation_function(calculate_weighted_sum(features, weights_), threshold_))] for features in
                            test_input_features]
        accuracy = calculate_accuracy(test_predictions, test_labels)
        print("Epoch ", epoch + 1, " accuracy:", accuracy)

    return weights_, threshold_, label_mapping_


def predict_observation(updated_weights, threshold_, label_mapping_):
    while True:
        input_observation = input("Provide the observation (numbers should be coma-separated): ")
        input_features = [float(x) for x in input_observation.split(',')]

        y = calculate_weighted_sum(input_features, updated_weights)
        prediction = activation_function(y, threshold_)

        for key, val in label_mapping_.items():
            if val == prediction:
                print("Predicted class:", key)

        user_input = input("Do you want to enter another observation? (1 - yes, 2 - exit the program): ")
        if user_input.lower() != "1":
            break


train_file = input("Provide the pass to the training file: ")
test_file = input("Provide the pass to the test file: ")
learning_rate = float(input("Provide the learning rate: "))
while True:
    num_of_epochs = int(input("Provide the number of epochs: "))
    if num_of_epochs > 0:
        break
    else:
        print("Number cannot be zero or less than zero. Please try again.")

weights, threshold, label_mapping = train_perceptron(learning_rate, num_of_epochs, train_file, test_file)
predict_observation(weights, threshold, label_mapping)
