import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load data
train_df = pd.read_csv("data/cleaned_train_features.csv")
test_df = pd.read_csv("data/cleaned_train_features.csv")
train_df['Personality'] = train_df['Personality'].map({'Introvert': 0, 'Extrovert': 1})
X = train_df.drop(columns=["Personality"])
y = train_df["Personality"]

# Preprocessing
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
])

X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(test_df)

X_train, X_val, y_train, y_val = train_test_split(X_processed, y.values, test_size=0.2, random_state=42)

# Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)

# Dataset
class PersonalityDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = PersonalityDataset(X_train_tensor, y_train_tensor)
val_dataset = PersonalityDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Model class
class PersonalityNet(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[1], 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# Hyperparameter tuning space
hidden_sizes_list = [(128, 128), (256, 128)]
dropouts = [0.05, 0.1]
learning_rates = [0.0005, 0.001]
weight_decays = [0.0001]

best_acc = 0
best_model = None
best_params = {}

# Hyperparameter tuning with early stopping & LR scheduler
for hidden_sizes in hidden_sizes_list:
    for dropout_rate in dropouts:
        for lr in learning_rates:
            for wd in weight_decays:
                print(f"\nTraining config: layers={hidden_sizes}, dropout={dropout_rate}, lr={lr}, wd={wd}")

                model = PersonalityNet(X_train.shape[1], hidden_sizes, dropout_rate)
                criterion = nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

                best_epoch_acc = 0
                patience = 10
                no_improve_epochs = 0

                for epoch in range(100):
                    model.train()
                    for xb, yb in train_loader:
                        pred = model(xb)
                        loss = criterion(pred, yb)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    scheduler.step()

                    model.eval()
                    with torch.no_grad():
                        val_pred = model(X_val_tensor)
                        val_acc = ((val_pred > 0.5).float() == y_val_tensor).float().mean().item()

                    print(f"Epoch {epoch+1}, Val Accuracy: {val_acc:.4f}")

                    if val_acc > best_epoch_acc:
                        best_epoch_acc = val_acc
                        best_model_state = model.state_dict()
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
                        if no_improve_epochs >= patience:
                            print("⏹️ Early stopping")
                            break

                if best_epoch_acc > best_acc:
                    best_acc = best_epoch_acc
                    best_model = PersonalityNet(X_train.shape[1], hidden_sizes, dropout_rate)
                    best_model.load_state_dict(best_model_state)
                    best_params = {
                        'hidden_sizes': hidden_sizes,
                        'dropout': dropout_rate,
                        'learning_rate': lr,
                        'weight_decay': wd
                    }

print("\n✅ Best Validation Accuracy:", best_acc)
print("✅ Best Parameters:", best_params)

# Final prediction
best_model.eval()
with torch.no_grad():
    test_pred = best_model(X_test_tensor)
    test_labels = (test_pred > 0.5).int().squeeze().numpy()

submission = pd.DataFrame({
    "id": range(18524, 18524 + len(test_labels)),
    "Personality": np.where(test_labels == 1, "Extrovert", "Introvert")
})

submission.to_csv("submission_nn_best_advanced_cleaning.csv", index=False)
