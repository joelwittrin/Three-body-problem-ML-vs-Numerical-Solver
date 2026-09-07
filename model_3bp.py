import numpy as np
import torch
import torch.nn as nn
from datagenerator_3bp import generate_training_data
from classifier import check_collision, check_escape, check_bounded

'''
This file initializes the model used to predict the three outcomes. Make sure the files datagenerator_3bp and classifier are imported correctly.
Further work will be done.
'''

def build_labels(collision_lst, escape_lst):

    # Creates labels according to indexes: 0 = collision, 1 = escape, 2 = bounded

    labels = []

    for i in range(len(collision_lst)):
        if collision_lst[i] == True:
            labels.append(0)
        elif escape_lst[i] == True:
            labels.append(1)
        else:
            labels.append(2)

    return labels

def get_training_data(mode="saved", n=500):

    # Generates new data or saved data. Using saved data is recommended if the model is adjusted and solely performance is studied.

    if mode == "new":
        x, y = generate_training_data(n=n)

        collision_lst = check_collision(y)
        escape_lst = check_escape(y)
        labels = build_labels(collision_lst, escape_lst)

        np.save('x_data.npy', x)
        np.save('y_data.npy', y)
        np.save('labels_data.npy', np.array(labels))

    elif mode == "saved":
        x = np.load('x_data.npy')
        y = np.load('y_data.npy')
        labels = np.load('labels_data.npy').tolist()

    else:
        raise ValueError("mode must be 'new' or 'saved'")

    return x, y, labels


# Only change to the second line if saved data is used! Choose between "saved" or "new ". Running both at the same time will double the runtime!
x, y, labels = get_training_data(mode="new", n=500) 
# x, y, labels = get_training_data(mode="saved")       

X = torch.tensor(x, dtype = torch.float32)
Y = torch.tensor(labels, dtype = torch.long)

# Normalize the data:
X_mean = X.mean(dim=0)
X_std = X.std(dim=0)
X_norm = (X - X_mean) / X_std


class TrajectoryModel(nn.Module):

    def __init__(self):
        super(TrajectoryModel, self).__init__()

        # Initialize three layers yielding the output labels. Note that it takes the full y-vector as input
        self.fullyconnected = nn.Sequential(
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.fullyconnected(x)

model = TrajectoryModel()


# Initialize for loss tracking
criteria = nn.CrossEntropyLoss()

# Utilize the Adam algorithm for optimization as this requires zero coding ourselves. Choose appropriate learning rate lr.
optimize = torch.optim.Adam(model.parameters(), lr=0.01)

# Due to imbalance in generated data, all classes are weighed as to force the model to learn
class_counts = np.bincount(labels)
weights = 1.0 / class_counts
weights = torch.tensor(weights, dtype=torch.float32)

criteria = nn.CrossEntropyLoss(weight=weights)


epochs = 500
for i in range(epochs):

    optimize.zero_grad()

    predictions = model(X_norm)
    loss = criteria(predictions, Y)
    loss.backward()
    optimize.step()

    if (i + 1) % 10 == 0:
        print(f"Epoch {i+1}/{epochs}, Loss: {loss.item():.6f}")


model.eval()

with torch.no_grad():

    # Generade test data to evaluate the model.
    x_test, y_test = generate_training_data(n=500)

    collision_test = check_collision(y_test)
    escape_test = check_escape(y_test)
    labels_test = build_labels(collision_test, escape_test)

    X_test = torch.tensor(x_test, dtype=torch.float32)
    Y_test = torch.tensor(labels_test, dtype=torch.long)

    # Normalize the test data with the same standard deviation as in training
    X_test_norm = (X_test - X_mean) / X_std

    outputs = model(X_test_norm)
    predicted = torch.argmax(outputs, dim=1)

    accuracy = (predicted == Y_test).float().mean()
    print(f"\nTest accuracy: {accuracy:.2%}")

    class_names = ['Collision', 'Escape', 'Bounded']
    for cls in range(3):
        mask = (Y_test == cls)
        if mask.sum() > 0:
            cls_acc = (predicted[mask] == Y_test[mask]).float().mean()
            print(f"{class_names[cls]}: {cls_acc:.2%} ({mask.sum().item()} samples)")
        else:
            print(f"{class_names[cls]}: inga testsamples i denna klass")


def debug():
    print(np.bincount(labels))