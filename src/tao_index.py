import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import joblib
from src.components import calculate_ks_distance, compute_mmd, calculate_dunns_index, calculate_silhouette_score, \
    calculate_davies_bouldin_index
from src.utils import z_normalize
import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# data loading
df_TEP = pd.read_csv('Data/TEP_Faulty_Training.csv')
df_TEP_normal = pd.read_csv('Data/TEP_FaultFree_Training.csv')

# Step 1: Filter and resample the faulty data to balance with the fault-free data
fault_types = df_TEP['faultNumber'].unique()
resampled_dfs = []
for fault in fault_types:
    fault_df = df_TEP[df_TEP['faultNumber'] == fault]
    # Sample 5% of each fault type data
    resampled = fault_df.sample(frac=0.05)
    resampled_dfs.append(resampled)

# Step 2: Combine all resampled faulty data with the fault-free data
fault_free_data = df_TEP_normal
combined_df = pd.concat(resampled_dfs + [fault_free_data], ignore_index=True)

# Step 3: Create a 3D structure separately for each fault type and then combine
unique_runs = combined_df['faultNumber'].unique()
feature_columns = combined_df.columns[3:55].tolist()

time_series_3d_list = []
for run in unique_runs:
    run_df = combined_df[combined_df['faultNumber'] == run]
    run_pivot = run_df.pivot(index='simulationRun', columns='sample', values=feature_columns)
    run_pivot.fillna(0, inplace=True)  # Fill any missing data points with 0
    run_pivot.columns = ['_'.join(map(str, col)).strip() for col in run_pivot.columns.values]
    run_pivot['faultNumber'] = run
    time_series_3d_list.append(run_pivot)

# Combine all pivoted data
time_series_3d = pd.concat(time_series_3d_list)

# Step 4: Convert multiclass to binary class
time_series_3d['faultNumber'] = time_series_3d['faultNumber'].apply(lambda x: 0 if x == 0 else 1)

# Prepare the data for DataLoader (example)
features = np.array([time_series_3d.drop('faultNumber', axis=1).values])
labels = np.array(time_series_3d['faultNumber'].values)

# concatenate the normal and faulty data into one dataframe
whole_df = pd.concat([df_TEP] + [df_TEP_normal], ignore_index=True)

# Create a 3D structure separately for each fault type and then combine them
unique_runs = whole_df['faultNumber'].unique()
indices = list(range(3, 25)) + list(range(44, df_TEP.shape[1]))
feature_columns = combined_df.columns[indices].tolist()
time_series_3d_list = []
for run in unique_runs:
    run_df = whole_df[whole_df['faultNumber'] == run]
    run_pivot = run_df.pivot(index='simulationRun', columns='sample', values=feature_columns)
    run_pivot.columns = ['_'.join(map(str, col)).strip() for col in run_pivot.columns.values]
    run_pivot['faultNumber'] = run
    time_series_3d_list.append(run_pivot)
# Combine all pivoted data
time_series_3d = pd.concat(time_series_3d_list)
# Convert multiclass to binary class
features = np.array([time_series_3d.drop('faultNumber', axis=1).values])
labels = np.array(time_series_3d['faultNumber'].values)
num_samples = features.shape[1]
reshaped_features = features.reshape(num_samples, 500, 33)
reduced_features = reshaped_features[:, :500, :]

# Normalize the data
normalized_data = z_normalize(reduced_features)

# split into training validation and test sets
X_train, X_val, y_train, y_val = train_test_split(normalized_data, labels, test_size=0.2, random_state=42)

# Now the DataLoader is ready to be used for training
# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

class TimeSeriesFCN(nn.Module):
    def __init__(self, num_classes, num_dim):
        super(TimeSeriesFCN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=num_dim, out_channels=128, kernel_size=8, padding='same'),
            nn.InstanceNorm1d(128),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.InstanceNorm1d(256),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same'),
            nn.InstanceNorm1d(128),
            nn.ReLU()
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)  # Number of classes defined by the problem

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_avg_pool(x)
        feature_extracted = x  # Intermediate output after global average pooling
        x = x.view(x.size(0), -1)  # Flatten the output
        logits = self.fc(x)
        return logits, feature_extracted

def train_and_evaluate_model(model, train_loader, val_loader, optimizer, criterion, epochs=100, patience=10):
    global best_model_wts
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc='Epochs'):
        # Training phase
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)  # Move data to the device
            optimizer.zero_grad()
            output,_ = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)  # Move data to the device
                output,_ = model(data)
                loss = criterion(output, target)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch}")
                break

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    model.to(device)  # Ensure model is on the right device
    return model, train_losses, val_losses

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesFCN(num_classes=21, num_dim=33).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the model
model, train_losses, val_losses = train_and_evaluate_model(model, train_loader, val_loader, optimizer, criterion, epochs=2000, patience=40)

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()
plt.show()

# save fcn models
torch.save(model.state_dict(), 'Models/fcn_tep2.pth')

# load fcn models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesFCN(num_classes=21,num_dim=33).to(device)
model.load_state_dict(torch.load('Models/fcn_tep2.pth'))

# calculate the prediction accuracy of the normal class and the 20 faulty classes
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)  # Ensure data is on the correct device
        outputs, _ = model(data)  # Get raw logits from the model
        _, predicted = torch.max(outputs, 1)  # Get the predicted class index
        all_predictions.extend(predicted.view(-1).cpu().numpy())  # Move predictions to CPU and convert to numpy
        all_targets.extend(target.view(-1).cpu().numpy())  # Move true labels to CPU and convert to numpy

# Calculate accuracy
accuracy = accuracy_score(all_targets, all_predictions)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Generate confusion matrix
cm = confusion_matrix(all_targets, all_predictions)

# Calculate per-class accuracy
class_accuracy = 100 * cm.diagonal() / cm.sum(axis=1)

# visualisation of the per-class accuracy
class_labels = range(len(class_accuracy))  # or use a list of names if you have them

plt.figure(figsize=(10, 5))
plt.bar(class_labels, class_accuracy, color='dodgerblue')
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.title('Per-Class Accuracy')
plt.xticks(class_labels)  # Ensure that class labels are properly displayed
plt.show()

# generate 210 datasets with any of the two classes and the corresponding target value
# normalized_data is feature and labels are the class labels
x_all = normalized_data  # shape (10500, 500, 33)
y_all = labels  # length 10500

# Calculate all 2-combinations of classes
class_combinations = list(itertools.combinations(range(21), 2))
print(f"Total combinations: {len(class_combinations)}")  # Should print 210

# Create lists to hold the datasets and accuracies
datasets = []
target = []

for combo in class_combinations:
    # Filter data for the two classes
    class_indices = np.where((y_all == combo[0]) | (y_all == combo[1]))[0]
    x_subset = x_all[class_indices]
    y_subset = y_all[class_indices]

    # Calculate average accuracy for these two classes
    avg_accuracy = (class_accuracy[combo[0]] + class_accuracy[combo[1]]) / 2
    target.append(avg_accuracy)

    # Store the dataset
    datasets.append((x_subset, y_subset))

# Print example information about the first dataset
print(f"First dataset shape: {datasets[0][0].shape}, {datasets[0][1].shape}")
print(f"First dataset average accuracy: {target[0]}")

# normalized_data is feature and labels are the class labels
x_all = normalized_data  # shape (10500, 500, 33)
y_all = labels  # length 10500

# Calculate all 2-combinations of classes
class_combinations = list(itertools.combinations(range(21), 2))

# Create lists to hold the datasets and accuracies
datasets = []
targets = []

proportions = np.linspace(0.1, 0.9, 9)  # 9 proportions from 0.1 to 0.9 inclusive
subset_size = 200  # Each subset should have 200 data points
num_repeats = 5  # Number of subsets to generate for each proportion

for combo in class_combinations:
    # Filter data for the two classes
    class_0_indices = np.where(y_all == combo[0])[0]
    class_1_indices = np.where(y_all == combo[1])[0]

    for proportion in proportions:
        for _ in range(num_repeats):
            num_class_0 = int(subset_size * proportion)
            num_class_1 = subset_size - num_class_0

            x_class_0_subset = x_all[np.random.choice(class_0_indices, num_class_0, replace=False)]
            x_class_1_subset = x_all[np.random.choice(class_1_indices, num_class_1, replace=False)]

            y_class_0_subset = y_all[np.random.choice(class_0_indices, num_class_0, replace=False)]
            y_class_1_subset = y_all[np.random.choice(class_1_indices, num_class_1, replace=False)]

            x_subset = np.concatenate((x_class_0_subset, x_class_1_subset), axis=0)
            y_subset = np.concatenate((y_class_0_subset, y_class_1_subset), axis=0)
            target_value = (class_accuracy[combo[0]] + class_accuracy[combo[1]]) / 2

            # Shuffle the subset to mix class 0 and class 1 data points
            shuffled_indices = np.random.permutation(len(y_subset))
            x_subset = x_subset[shuffled_indices]
            y_subset = y_subset[shuffled_indices]

            datasets.append((x_subset, y_subset))
            targets.append(target_value)

print(f"Total combinations: {len(targets)}")  # Should print 9450
# Print example information about the first dataset
print(f"First dataset shape: {datasets[0][0].shape}, {datasets[0][1].shape}")
print(f"First dataset target value: {targets[0]}")

# use the feature extractor to map the data to the latent space
datasets_latent = []
for i in range(len(datasets)):
    reshaped_features,resampled_labels = datasets[i]
    data_loader = DataLoader(TensorDataset(torch.tensor(reshaped_features).permute(0, 2, 1).float()), batch_size=64, shuffle=False)
    latent_features = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch_data = batch[0].to(device)
            _, latent = model(batch_data)
            latent = latent.view(latent.size(0), -1)
            latent = latent.cpu().numpy()  # Move each batch's latent features to CPU and convert to numpy
            latent_features.append(latent)

    # Concatenate all batch results into a single numpy array
    latent_features = np.concatenate(latent_features, axis=0)
    datasets_latent.append((latent_features,resampled_labels))

# calculate the metadata for the subsets
metadata_list = []
for i in range(len(datasets_latent)):
    reshaped_features,y = datasets_latent[i]  # unpacks the dataset tuple
    # Remap the labels to 0 and 1
    unique_labels = np.unique(y)
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
    scaler = StandardScaler()
    X = scaler.fit_transform(reshaped_features)
    # calculate the meta
    # calculate the 9 separability metrics
    ks_distance, integral_distance = calculate_ks_distance('distance', X, y, k=5)
    ks_density, integral_density = calculate_ks_distance('density', X, y, k=5)
    ks_dimension, integral_dimension = calculate_ks_distance('dimension', X, y, k=5)
    ks_nlabels, integral_nlabels = calculate_ks_distance('nlabels', X, y, k=5)
    # MMD value of the whole dataset
    X0 = X[y == 0]
    X1 = X[y == 1]
    mmd_value = compute_mmd(X0, X1)
    # Dunn's index
    dunn_score = calculate_dunns_index(X, y)
    sil_score = calculate_silhouette_score(X,y)
    davies = calculate_davies_bouldin_index(X,y)
    X_meta = [ks_distance, integral_distance, ks_density, integral_density, ks_dimension, integral_dimension, ks_nlabels, integral_nlabels, mmd_value, dunn_score, sil_score, davies]
    # Convert the list to a 2D numpy array with one row
    X_meta_array = np.array([X_meta])
    metadata_list.append(X_meta_array)

# Concatenate the metadata of each subset into a single numpy array
metadata_tep = np.concatenate(metadata_list, axis=0)

# Convert to DataFrame
df_metadata = pd.DataFrame(metadata_tep)
# Add labels as the last column
df_metadata['Label'] = targets
# Save to CSV file
df_metadata.to_csv('Data/metadata_v2.csv', index=False)

# Load the tep metadata CSV file
df_metadata = pd.read_csv('Data/metadata_v2.csv')

# Separate the features and labels
metadata_tep = df_metadata.iloc[:, :-1].values  # All columns except the last
targets = df_metadata['Label'].tolist()  # The last column as a list

metadata_tep = metadata_tep[:, [0, 2, 4, 6, 8]] # only use the 5 component for the proposed separability metric

# Initialize the regression model
linear_model = LinearRegression()

# Standardizing the features
scaler_X = StandardScaler()
X_scaled_metadata_tep = scaler_X.fit_transform(metadata_tep)

# Ensure y is a numpy array and transfer to CPU if it's a CUDA tensor
if isinstance(targets, torch.Tensor):
    y = targets.cpu().numpy()
else:
    y = np.array(targets)

y = np.array([i / 100 for i in y])  # Normalize target

# Perform 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mse_scores = -cross_val_score(linear_model, metadata_tep, y, cv=kf, scoring='neg_mean_squared_error')

# Calculate mean and standard deviation of MSE
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)

print(f"Linear Regression MSE: {mean_mse:.5f} ± {std_mse:.5f}")

# Initialize the Random Forest model
random_forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Standardizing the features
scaler_X = StandardScaler()
X_scaled_metadata_tep = scaler_X.fit_transform(metadata_tep)

# Ensure y is a numpy array and transfer to CPU if it's a CUDA tensor
if isinstance(targets, torch.Tensor):
    y = targets.cpu().numpy()
else:
    y = np.array(targets)

y = np.array([i / 100 for i in y])  # Normalize target

# Perform 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mse_scores = -cross_val_score(random_forest_reg, metadata_tep, y, cv=kf, scoring='neg_mean_squared_error')

# Calculate mean and standard deviation of MSE
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)

print(f"Random Forest Regression MSE: {mean_mse:.5f} ± {std_mse:.5f}")

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(metadata_tep, targets, test_size=0.2, random_state=42)

# Ensure y_train and y_test are NumPy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# Standardizing the features (SVR is sensitive to feature scales)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# Hyper-parameter tuning using GridSearchCV
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1]
}

svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train_scaled)

# Best parameters found
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Train the model with 10-fold cross validation
best_svr = SVR(kernel=best_params['kernel'], C=best_params['C'], epsilon=best_params['epsilon'])
whole_scaled = scaler_X.fit_transform(metadata_tep)
target_scaled = scaler_y.fit_transform(np.array(targets).reshape(-1, 1)).ravel()

# Perform 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mse_scores = -cross_val_score(best_svr, whole_scaled, target_scaled, cv=kf, scoring='neg_mean_squared_error')

# Calculate mean and standard deviation of MSE
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)

print(f"SVR Regression MSE: {mean_mse:.5f} ± {std_mse:.5f}")

# Load the CSV file
df_metadata = pd.read_csv('Data/metadata_v2.csv')

# Separate the features and labels
metadata_tep = df_metadata.iloc[:, :-1].values  # All columns except the last
targets = df_metadata['Label'].tolist()  # The last column as a list
metadata_tep = metadata_tep[:, [0, 2, 4, 6, 8]]

# Standardizing the features
scaler_X = StandardScaler()
X_scaled_metadata_tep = scaler_X.fit_transform(metadata_tep)

# Ensure y is a numpy array and transfer to CPU if it's a CUDA tensor
if isinstance(targets, torch.Tensor):
    y = targets.cpu().numpy()
else:
    y = np.array(targets)

y = np.array([i / 100 for i in y])  # Normalize target

# training the rf model using the whole TEP dataset
random_forest_reg.fit(metadata_tep, y)
# Predict and calculate the mean squared error on the entire dataset (not part of cross-validation)
y_pred_rf = random_forest_reg.predict(metadata_tep)
mse_rf = mean_squared_error(y, y_pred_rf)
print(f"Random Forest Regression MSE on entire dataset: {mse_rf:.5f}")

# save the random forest model
joblib.dump(random_forest_reg, 'Models/random_forest_model5.pkl')

# train the masked models for ablation study
# load the random forest model
random_forest_reg = joblib.load('Models/random_forest_model5.pkl')

# Iterate through each column in metadata_tep
for col_idx in range(metadata_tep.shape[1]):
    # Mask the current column
    X_masked = np.delete(metadata_tep, col_idx, axis=1)

    # Initialize the regression model
    random_forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model using the remaining columns
    random_forest_reg.fit(X_masked, y)

    # Save the trained model
    model_filename = f'Models/random_forest_model_masked_col_{col_idx}.pkl'
    joblib.dump(random_forest_reg, model_filename)