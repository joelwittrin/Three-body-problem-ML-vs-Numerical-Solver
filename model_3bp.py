import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from datagenerator_3bp import generate_training_data
from classifier import check_collision, check_escape, check_bounded

'''
Trains a classifier (PyTorch) to predict trajectory outcome (collision/escape/bounded)
from the initial state of a three-body system.
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



def get_training_data(mode="saved", n=1000):

    # Generates new data or loads saved data. Using saved data is recommended
    # once the model is being tuned, so the solver is not re-run every time.
    # mode = "new" or "saved"

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


# Only change to "saved" once data has been generated at least once! Generate new data with "new"
x, y, labels = get_training_data(mode="new", n=10000)



x_train, x_test, labels_train, labels_test = train_test_split(
    x, labels, test_size=0.2, stratify=labels, random_state=42
)

X_train = torch.tensor(x_train, dtype=torch.float32)
X_test = torch.tensor(x_test, dtype=torch.float32)
Y_train = torch.tensor(labels_train, dtype=torch.long)
Y_test = torch.tensor(labels_test, dtype=torch.long)

# Normalize using ONLY training statistics, applied to both sets
X_mean = X_train.mean(dim=0)
X_std = X_train.std(dim=0)

X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std



class TrajectoryModel(nn.Module):

    def __init__(self, input_dim):
        super(TrajectoryModel, self).__init__()

        self.fullyconnected = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.fullyconnected(x)


model = TrajectoryModel(input_dim=X_train.shape[1])

class_counts = np.bincount(labels_train)
weights = 1.0 / class_counts
weights = weights / weights.sum() * len(class_counts)   # normalize, keep signal strong
weights = torch.tensor(weights, dtype=torch.float32)

criteria = nn.CrossEntropyLoss(weight=weights)

optimize = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimize, step_size=500, gamma=0.5)


### Trainig loop below ###


epochs = 2000
for i in range(epochs):

    model.train()
    optimize.zero_grad()

    predictions = model(X_train_norm)
    loss = criteria(predictions, Y_train)
    loss.backward()
    optimize.step()
    scheduler.step()

    if (i + 1) % 100 == 0:
        print(f"Epoch {i+1}/{epochs}, Loss: {loss.item():.6f}")



model.eval()

with torch.no_grad():

    # Accuracy on the training data
    train_predicted = torch.argmax(model(X_train_norm), dim=1)
    train_accuracy = (train_predicted == Y_train).float().mean()
    print(f"\nTrain accuracy: {train_accuracy:.2%}")

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
            print(f"{class_names[cls]}: no test samples for this class")

    cm = confusion_matrix(Y_test.numpy(), predicted.numpy())
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print("             Coll  Esc  Bound")
    for idx, row in enumerate(cm):
        print(f"{class_names[idx]:>10}: {row}")


def debug():
    print("Class distribution (train):", np.bincount(labels_train))
    print("Class distribution (test):", np.bincount(labels_test))