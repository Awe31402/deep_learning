import os
import deeplay as dl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from torch.nn import Sigmoid
from torch.nn import MSELoss

if not os.path.exists("MNIST_dataset"):
    os.system("git clone https://github.com/DeepTrackAI/MNIST_dataset")

train_path = os.path.join("MNIST_dataset", "mnist", "train")
trained_image_files = sorted(os.listdir(train_path))

print(len(trained_image_files))  # Output: 60000

trained_images = []
for file in trained_image_files:
    image = plt.imread(os.path.join(train_path, file))
    trained_images.append(image)

print(len(trained_images))  # Output: 60000
print(trained_images[0].shape)  # Output: (28, 28)

trained_digits = []
for file in trained_image_files:
    filename = os.path.basename(file)
    digit = int(filename[0])
    trained_digits.append(digit)

fig, axs = plt.subplots(nrows=3, ncols=10, figsize=(20, 6))
for ax in axs.ravel():
    idx = np.random.randint(0, len(trained_images))
    ax.imshow(trained_images[idx], cmap='gray')
    ax.set_title(f"Digit: {trained_digits[idx]}")
    ax.axis('off')

if matplotlib.get_backend().lower() in ['agg', 'pdf', 'svg', 'ps']:
    output_path = 'mnist_samples.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Non-interactive backend '{matplotlib.get_backend()}' detected. Saved figure to {os.path.abspath(output_path)}")
else:
    plt.show()

mlp_template = dl.MultiLayerPerceptron(
    in_features=28 * 28, hidden_features=[32, 32], out_features=10,
)
mlp_template[..., "activation"].configure(Sigmoid)
mlp_model = mlp_template.create()
print(mlp_model)

print(f"{sum(p.numel() for p in mlp_model.parameters())} trainable parameters")
classifier_template = dl.Classifier(
    model=mlp_model, loss=MSELoss(), optimizer=dl.SGD(lr=0.001), num_classes=10,
    make_targets_one_hot=True,
)

classifier = classifier_template.create()
print(classifier)

train_images_digits = list(zip(trained_images, trained_digits))
train_dataloader = dl.DataLoader(train_images_digits, shuffle=True)

trainer = dl.Trainer(max_epochs=1, accelerator="auto")

trainer.fit(classifier, train_dataloader)

test_path = os.path.join("MNIST_dataset", "mnist", "test")
test_images_files = sorted(os.listdir(test_path))

test_images, test_digits = [], []
for file in test_images_files:
    image = plt.imread(os.path.join(test_path, file))
    test_images.append(image)

    filename = os.path.basename(file)
    digit = int(filename[0])
    test_digits.append(digit)

test_images_digits = list(zip(test_images, test_digits))
test_dataloader = dl.DataLoader(test_images_digits, shuffle=False)
trainer.test(classifier, test_dataloader)