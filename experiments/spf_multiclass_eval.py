import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from src.components import calculate_ks_distance, compute_mmd, calculate_dunns_index, calculate_silhouette_score, \
    calculate_davies_bouldin_index, calculate_N2, calculate_N4, calculate_T1, calculate_LSC

# test on the benchmark datasets (SPF)
# load the random forest model
random_forest_reg = joblib.load('Models/random_forest_model5.pkl')
df = pd.read_csv('Data/steel-plate/faults.csv')
# Split features and labels
X = df.iloc[:, :27]  # the first 27 columns are features
classes = df.columns[27:]  # class labels are in the last seven columns

binary_datasets = []

for class_label in classes:
    # Creating a binary dataset for each class
    Y_bin = df[class_label]
    binary_datasets.append((class_label, (X, Y_bin)))

separability_list_rf = []
#separability_list_ml = []
N2_list = []
N4_list = []
T1_list = []
LSC_list = []
dunn_list = []
silhouette_list = []
davies_list = []
for dataset in binary_datasets:
    class_label, (X, y) = dataset  # unpacks the dataset tuple
    scaler = StandardScaler()
    X_steel = scaler.fit_transform(X)
    # calculate the meta
    # calculate the 9 separability metrics
    ks_distance, _ = calculate_ks_distance('distance', X_steel, y, k=5)
    ks_density, _ = calculate_ks_distance('density', X_steel, y, k=5)
    ks_dimension, _ = calculate_ks_distance('dimension', X_steel, y, k=5)
    ks_nlabels, _ = calculate_ks_distance('nlabels', X_steel, y, k=5)
    # MMD value of the whole dataset
    X0 = X_steel[y == 0]
    X1 = X_steel[y == 1]
    mmd_value = compute_mmd(X0, X1)
    dunn_score = calculate_dunns_index(X_steel, y)
    sil_score = calculate_silhouette_score(X_steel, y)
    davies = calculate_davies_bouldin_index(X_steel, y)
    #X_steel_meta = [ks_distance, ks_density, ks_dimension, ks_nlabels, mmd_value, dunn_score, sil_score, davies]
    X_steel_meta = [ks_distance, ks_density, ks_dimension, ks_nlabels, mmd_value]
    # predicted separability
    separability_pred_rf = random_forest_reg.predict([X_steel_meta])
    #separability_pred_ml = linear_model.predict([X_steel_meta])
    #predicted_value = random_forest_reg.predict([X_steel_meta])[0]
    #ratio = X0.shape[0] / X1.shape[0] if X0.shape[0] > X1.shape[0] else X1.shape[0] / X0.shape[0]
    #separability_pred = predicted_value / ratio

    N2 = calculate_N2(X_steel, y)
    N4 = calculate_N4(X_steel, y)
    T1 = calculate_T1(X_steel, y)
    LSC = calculate_LSC(X_steel, y)
    dunn_score = calculate_dunns_index(X_steel, y)
    silhouette = calculate_silhouette_score(X_steel, y)
    davies = calculate_davies_bouldin_index(X_steel, y)

    separability_list_rf.append(separability_pred_rf)
    #separability_list_ml.append(separability_pred_ml)
    N2_list.append(N2)
    N4_list.append(N4)
    T1_list.append(T1)
    LSC_list.append(LSC)
    dunn_list.append(dunn_score)
    silhouette_list.append(silhouette)
    davies_list.append(davies)

print(max(separability_list_rf),max(N2_list),max(N4_list),max(T1_list),max(LSC_list), max(dunn_list), max(silhouette_list), max(davies_list))

# masked results for ablation study
df = pd.read_csv('Data/steel-plate/faults.csv')
# Split features and labels
X = df.iloc[:, :27]  # the first 27 columns are features
classes = df.columns[27:]  # class labels are in the last seven columns

binary_datasets = []

for class_label in classes:
    # Creating a binary dataset for each class
    Y_bin = df[class_label]
    binary_datasets.append((class_label, (X, Y_bin)))

# Initialize the 2D array to store separability results
separability_matrix_rf = np.zeros((len(binary_datasets), 5))

for i, dataset in enumerate(binary_datasets):
    class_label, (X, y) = dataset  # unpacks the dataset tuple
    scaler = StandardScaler()
    X_steel = scaler.fit_transform(X)
    # calculate the meta
    # calculate the 9 separability metrics
    ks_distance, integral_distance = calculate_ks_distance('distance', X_steel, y, k=5)
    ks_density, integral_density = calculate_ks_distance('density', X_steel, y, k=5)
    ks_dimension, integral_dimension = calculate_ks_distance('dimension', X_steel, y, k=5)
    ks_nlabels, integral_nlabels = calculate_ks_distance('nlabels', X_steel, y, k=5)
    # MMD value of the whole dataset
    X0 = X_steel[y == 0]
    X1 = X_steel[y == 1]
    mmd_value = compute_mmd(X0, X1)
    X_steel_meta = [ks_distance, ks_density, ks_dimension, ks_nlabels, mmd_value]

    # predicted separability
    masked_result = []
    # Iterate through each column in metadata_tep
    for col_idx in range(len(X_steel_meta)):
        model_filename = f'Models/random_forest_model_masked_col_{col_idx}.pkl'
        # load the regression model
        random_forest_reg = joblib.load(model_filename)

        # Mask the current element (column)
        X_masked = X_steel_meta[:col_idx] + X_steel_meta[col_idx+1:]
        # predict the separability
        masked_result.append(random_forest_reg.predict([X_masked])[0])

    # Store the masked result in the separability matrix
    separability_matrix_rf[i, :] = masked_result

max_separability_rf = np.max(separability_matrix_rf, axis=0)
print(max_separability_rf)