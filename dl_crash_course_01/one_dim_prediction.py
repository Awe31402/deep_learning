from loader import load_data_1d
from numpy.random import default_rng
import matplotlib.pyplot as plt

def plot_data(x, y_gt):
    plt.clf()
    plt.scatter(x, y_gt, color='blue', label='Ground Truth')
    plt.xlabel('Feature')
    plt.ylabel('Label')
    plt.title('1D Dataset')
    plt.legend()
    plt.savefig('ground-truth.png')

def plot_predictions(x, y_gt, y_pred, filename='predictions.png'):
    plt.clf()
    plt.scatter(x, y_gt, color='blue', label='Ground Truth')
    plt.scatter(x, y_pred, color='red', label='Predictions', marker='x')
    plt.xlabel('Feature')
    plt.ylabel('Label')
    plt.title('1D Dataset with Predictions')
    plt.legend()
    plt.savefig(filename)

def neuron_class_1d(w0, x):
    return (w0 * x > 0).astype(int)

if __name__ == "__main__":
    dataset_to_use = 'noisy'  # Change to 'noisy' for the noisy dataset
    train_filename = f'data_class_1d_{dataset_to_use}.csv'
    test_filename = f'data_class_1d_{dataset_to_use}_test.csv'

    (x, y_gt) = load_data_1d(train_filename)
    plot_data(x, y_gt)

    rng = default_rng()
    w0 = rng.normal()
    print(f"Initial weight: {w0}")
    y_pred = neuron_class_1d(w0, x)
    plot_predictions(x, y_gt, y_pred, filename='initial-predictions.png')

    num_samples = len(x)
    num_epochs = 1000
    learning_rate = 0.01

    # Training loop
    for epoch in range(num_epochs):
        selected_index = rng.integers(0, num_samples)
        x_i = x[selected_index]
        y_i = y_gt[selected_index]
        y_pred_i = neuron_class_1d(w0, x_i)
        error = y_i - y_pred_i
        w0 += learning_rate * error * x_i

    print(f"Trained weight: {w0}")
    y_pred = neuron_class_1d(w0, x)
    plot_predictions(x, y_gt, y_pred, filename='trained-predictions.png')

    # Verify that the trained model with test data
    (x_test, y_test) = load_data_1d(test_filename)
    y_test_pred = neuron_class_1d(w0, x_test)
    accuracy = (y_test_pred == y_test).mean()
    print(f"Test Accuracy: {accuracy:.2f}")