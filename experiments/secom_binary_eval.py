import pandas as pd
from sklearn.preprocessing import StandardScaler
from feature_engine.selection import DropConstantFeatures, SmartCorrelatedSelection
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import joblib
from src.components import calculate_ks_distance, compute_mmd, calculate_dunns_index, calculate_silhouette_score, \
    calculate_davies_bouldin_index, calculate_N2, calculate_N4, calculate_T1, calculate_LSC

# test on the benchmark datasets (SECOM)
# load the random forest model
random_forest_reg = joblib.load('Models/random_forest_model5.pkl')
# test the model performance on benchmark datasets
df = pd.read_csv('Data/uci-secom/uci-secom.csv')
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

y = y.replace({-1: 0})

seed = 42

X = X.sample(frac=1, random_state = seed)
y = y.loc[X.index]

X_copy = X.copy()
X_copy = DropConstantFeatures(missing_values='ignore').fit_transform(X_copy)

X_copy = SmartCorrelatedSelection(threshold=0.8, selection_method='missing_values').fit_transform(X_copy)

miss_series = X_copy.isnull().sum() / len(X_copy)
miss_series.sort_values(ascending = False).head(22)
drop_miss = miss_series.loc[miss_series > 0.10]
X_copy = X_copy.drop(drop_miss.index, axis = 1)

# Fit the pipeline to the feature-reduced dataset
pipe_lr = Pipeline(steps = [
    ('scaler', MinMaxScaler()),
    ('imputer', KNNImputer(n_neighbors = 40))
])

pipe_lr.fit(X_copy, y)

# Transform the feature-reduced data to get the imputed (and scaled) dataset
X_imputed = pipe_lr.transform(X_copy)

# Get the DataFrame back with column names from X_copy
X_imputed_df = pd.DataFrame(X_imputed, columns=X_copy.columns, index=X_copy.index)
scaler = StandardScaler()
X_secom = scaler.fit_transform(X_imputed_df)
# calculate the meta
# calculate the 4 separability metrics
ks_distance, _ = calculate_ks_distance('distance', X_secom, y, k=5)
ks_density, _ = calculate_ks_distance('density', X_secom, y, k=5)
ks_dimension, _ = calculate_ks_distance('dimension', X_secom, y, k=5)
ks_nlabels, _ = calculate_ks_distance('nlabels', X_secom, y, k=5)
# MMD value of the whole dataset
X0 = X_secom[y == 0]
X1 = X_secom[y == 1]
mmd_value = compute_mmd(X0, X1)
dunn_score = calculate_dunns_index(X_secom, y)
sil_score = calculate_silhouette_score(X_secom, y)
davies = calculate_davies_bouldin_index(X_secom, y)
#X_secom_meta = [ks_distance, ks_density, ks_dimension, ks_nlabels, mmd_value, dunn_score, sil_score, davies]
X_secom_meta = [ks_distance, ks_density, ks_dimension, ks_nlabels, mmd_value]
print(f'The RF model metric value is {random_forest_reg.predict([X_secom_meta])[0]}')
#print(f'The Multi-linear model metric value is {linear_model.predict([X_secom_meta])[0]}')
N2 = calculate_N2(X_secom, y)
N4 = calculate_N4(X_secom, y)
T1 = calculate_T1(X_secom, y)
LSC = calculate_LSC(X_secom, y)
Dunn = calculate_dunns_index(X_secom, y)
print(N2,N4,T1,LSC,Dunn)

# masked results for ablation study
# test the model performance on benchmark datasets
df = pd.read_csv('Data/uci-secom/uci-secom.csv')
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

y = y.replace({-1: 0})

seed = 42

X = X.sample(frac=1, random_state = seed)
y = y.loc[X.index]

X_copy = X.copy()
X_copy = DropConstantFeatures(missing_values='ignore').fit_transform(X_copy)

X_copy = SmartCorrelatedSelection(threshold=0.8, selection_method='missing_values').fit_transform(X_copy)

miss_series = X_copy.isnull().sum() / len(X_copy)
miss_series.sort_values(ascending = False).head(22)
drop_miss = miss_series.loc[miss_series > 0.10]
X_copy = X_copy.drop(drop_miss.index, axis = 1)

# Fit the pipeline to the feature-reduced dataset
pipe_lr = Pipeline(steps = [
    ('scaler', MinMaxScaler()),
    ('imputer', KNNImputer(n_neighbors = 40))
])

pipe_lr.fit(X_copy, y)

# Transform the feature-reduced data to get the imputed (and scaled) dataset
X_imputed = pipe_lr.transform(X_copy)

# Get the DataFrame back with column names from X_copy
X_imputed_df = pd.DataFrame(X_imputed, columns=X_copy.columns, index=X_copy.index)
scaler = StandardScaler()
X_secom = scaler.fit_transform(X_imputed_df)

masked_result = []
# calculate the 9 separability metrics
ks_distance, integral_distance = calculate_ks_distance('distance', X_secom, y, k=5)
ks_density, integral_density = calculate_ks_distance('density', X_secom, y, k=5)
ks_dimension, integral_dimension = calculate_ks_distance('dimension', X_secom, y, k=5)
ks_nlabels, integral_nlabels = calculate_ks_distance('nlabels', X_secom, y, k=5)
# MMD value of the whole dataset
X0 = X_secom[y == 0]
X1 = X_secom[y == 1]
mmd_value = compute_mmd(X0, X1)
X_secom_meta_complete = [ks_distance, ks_density, ks_dimension, ks_nlabels, mmd_value]
# Iterate through each column in metadata_tep
for col_idx in range(len(X_secom_meta_complete)):
    model_filename = f'Models/random_forest_model_masked_col_{col_idx}.pkl'
    # load the regression model
    random_forest_reg = joblib.load(model_filename)

    # Mask the current element (column)
    X_masked = X_secom_meta_complete[:col_idx] + X_secom_meta_complete[col_idx+1:]
    # predict the separability
    masked_result.append(random_forest_reg.predict([X_masked])[0])
print(masked_result)