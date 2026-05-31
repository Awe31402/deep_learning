from loader import load_data_2d
from numpy.random import default_rng
import matplotlib.pyplot as plt

def plot_data_2d(x, y, label='ground-truth'):
    plt.clf()
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr', label=label)
    plt.colorbar()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2D Dataset')
    plt.legend()
    plt.savefig(f'{label}.png')

def neuron_class_2d(w, x):
    return ((w[0] * x[:, 0] + w[1] * x[:, 1]) > 0).astype(int)

if __name__ == "__main__":
    dataset_to_use = 'noisy'  # Change to 'noisy' for the noisy dataset
    train_filename = f'data_class_2d_{dataset_to_use}.csv'
    test_filename = f'data_class_2d_{dataset_to_use}_test.csv'

    (x, y_gt) = load_data_2d(train_filename)
    print(x)
    print(y_gt)
    plot_data_2d(x, y_gt, label='ground-truth')

    rng = default_rng()
    w = rng.normal(size=2)

    print(f"Initial weights: {w}")
    y_pred = neuron_class_2d(w, x)
    plot_data_2d(x, y_pred, label='initial-predictions')

    # Training loop
    num_samples = len(x)
    num_epochs = 10000
    learning_rate = 0.01

    for epoch in range(num_epochs):
        selected_index = rng.integers(0, num_samples)
        x_i = x[selected_index]
        y_i = y_gt[selected_index]
        y_pred_i = neuron_class_2d(w, x_i.reshape(1, -1))[0]
        error = y_i - y_pred_i
        w += learning_rate * error * x_i

    print(f"Trained weights: {w}")
    y_pred = neuron_class_2d(w, x)
    plot_data_2d(x, y_pred, label='trained-predictions')
    print(x)
    print(y_pred)

    # verify that the trained model with test data
    (x_test, y_test) = load_data_2d(test_filename)
    y_test_pred = neuron_class_2d(w, x_test)
    accuracy = (y_test_pred == y_test).mean()
    print(f"Test Accuracy: {accuracy:.2f}")    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Test Accuracy: {accuracy:.2f}")
