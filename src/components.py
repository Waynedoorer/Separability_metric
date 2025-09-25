from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
from math import sqrt, exp
from scipy.spatial import distance_matrix
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import qmc
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

# utility function to add noise to the labels of the dataset
def add_noise_to_labels(labels, noise_mean, noise_sd):
    """Add random noise to labels and return new labels and mean noise value."""
    noise = np.random.normal(noise_mean, noise_sd, size=labels.shape)
    new_labels = np.where(noise + labels >= 0.5, 1, 0)
    return new_labels, noise.mean()

def flip_labels_randomly(labels, sigma):
    """Flip labels for a percentage (sigma) of randomly selected data points in each class."""
    flipped_labels = labels.copy()

    for label_class in np.unique(labels):
        class_indices = np.where(labels == label_class)[0]
        n_flip = int(len(class_indices) * sigma)  # Number of labels to flip in each class
        indices_to_flip = np.random.choice(class_indices, size=n_flip, replace=False)

        # Flip the labels
        flipped_labels[indices_to_flip] = 1 - flipped_labels[indices_to_flip]

    return flipped_labels

# the original space: distance-based metric + K-Means clustering algorithm
def calculate_kmeans_metric(data, true_labels):
    """Calculate K-Means metric (Silhouette Score)."""
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(data)
    labels = kmeans.labels_
    return adjusted_rand_score(true_labels, labels)

def calculate_local_distance(data, k=5):
    """Calculate average local distance for each data point based on k-nearest neighbors."""
    if len(data) <= k:
        # If there are not enough data points, reduce k to the number of available data points
        k = len(data) - 1
        if k <= 0:
            # If there's only one data point, return a default low density
            return np.array([0.001] * len(data))

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
    distances, _ = nbrs.kneighbors(data)

    # Exclude the distance to the point itself (first column) and take the mean of the remaining distances
    local_distance = np.mean(distances[:, 1:], axis=1)
    return local_distance

# the original space: density-based metric + GMM clustering algorithm
def calculate_local_density(data, k=5):
    """Calculate modified local density for each data point based on mean and variance of k-nearest neighbors distances."""
    if len(data) <= k:
        k = len(data) - 1
        if k <= 0:
            return np.array([0.001] * len(data))

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
    distances, _ = nbrs.kneighbors(data)

    # Calculate mean and variance of the distances (excluding the distance to the point itself)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    variance_distances = np.std(distances[:, 1:], axis=1)

    # Combine mean and std into a modified density metric
    # This emphasizes areas of both high density (low mean distance) and low dispersion (low variance)
    modified_local_density = 1 / (mean_distances * (1 + variance_distances) + 0.0001)

    return modified_local_density

def cluster_with_gmm(data, n_clusters=2, k=5):
    """Cluster data using Gaussian Mixture Models based on local density metrics."""
    # Calculate local density
    local_density = calculate_local_density(data,k)

    # Apply GMM to the local density metrics
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    gmm.fit(local_density.reshape(-1, 1))  # Reshape for GMM
    labels = gmm.predict(local_density.reshape(-1, 1))

    return labels

def calculate_density_metric(data, true_labels, k=5):
    """Calculate density metric (Adjusted Rand Score)."""
    labels = cluster_with_gmm(data,k)
    return adjusted_rand_score(true_labels, labels)

# the original space: dimension-based metric + GMM clustering algorithm
def mle_lid(data, k=5):
    """
    Compute the LID values for each point in the dataset using the MLE approach.

    Parameters:
    data (numpy.ndarray): The dataset for which LID needs to be computed.
    k (int): The number of nearest neighbors to consider for each point.

    Returns:
    numpy.ndarray: An array of LID values for each point in the dataset.
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(data)
    distances, _ = nbrs.kneighbors(data)

    # Exclude the distance to the point itself (distance = 0)
    distances = distances[:, 1:]

    # Compute the LID estimate for each point
    lid_estimates = -k / np.sum(np.log(distances / distances[:, -1, None]), axis=1)

    return lid_estimates

def calculate_lid_gmm_metric(data, true_labels):
    """Calculate LID + GMM metric (Adjusted Rand Score)."""
    # LID calculation
    lid_estimates = mle_lid(data, k=20)
    lid_values_2d = lid_estimates.reshape(-1, 1)  # Reshape to 2D array
    # Here, we simply use Gaussian Mixture Model (GMM) for demonstration.
    gmm = GaussianMixture(n_components=2, random_state=0).fit(lid_values_2d)
    labels = gmm.predict(lid_values_2d)
    return adjusted_rand_score(true_labels, labels)

# neighbourhood labels
def calculate_local_nl(data, true_labels, k=5):
    # Convert labels from {0, 1} to {-1, 1}
    mapped_labels = np.where(true_labels == 0, -1, 1)
    """Calculate modified local density for each data point based on mean and variance of k-nearest neighbors distances."""
    if len(data) <= k:
        k = len(data) - 1
        if k <= 0:
            return np.array([0.001] * len(data))
    # Initialize the nearest neighbors finder
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
    distances, indices = nbrs.kneighbors(data)

    # Calculate the local metric for each point
    local_metrics = []
    for idx, neighbors in enumerate(indices):
        # Exclude the point itself (first index is the point itself)
        neighbor_labels = mapped_labels[neighbors[1:]]
        local_metric = np.abs(np.mean(neighbor_labels))
        local_metrics.append(local_metric)

    return np.array(local_metrics)

# local discrepancy
def calculate_local_discrepancy(data):
    # Normalize the entire dataset to the unit hyper-cube
    l_bounds = np.min(data, axis=0)
    u_bounds = np.max(data, axis=0)
    normalized_data = qmc.scale(data, l_bounds, u_bounds)

    # Calculate the overall discrepancy
    overall_discrep = qmc.discrepancy(normalized_data)

    # Calculate local discrepancies and their differences
    local_metrics = []
    for i in range(len(data)):
        # Exclude the current point
        reduced_data = np.delete(normalized_data, i, axis=0)
        local_discrep = qmc.discrepancy(reduced_data)

        # Calculate the difference in discrepancies
        discrepancy_difference = overall_discrep - local_discrep
        local_metrics.append(discrepancy_difference)

    return np.array(local_metrics)

def gaussian_kernel(x, y, sigma):
    """Compute the Gaussian (RBF) kernel between x and y."""
    gamma = 1.0 / (2 * sigma**2)
    return np.exp(-gamma * euclidean_distances(x, y, squared=True))

def median_heuristic(x, y):
    """Compute the median of pairwise distances for use in the Gaussian kernel."""
    combined = np.vstack([x, y])
    distances = euclidean_distances(combined, squared=True)
    return np.sqrt(0.5 * np.median(distances[distances > 0]))  # Median of the non-zero distances

def compute_mmd(x, y, sigma=None):
    """Compute the Maximum Mean Discrepancy (MMD) with adjustments for imbalance."""
    if sigma is None:
        sigma = median_heuristic(x, y)

    n = len(x)
    m = len(y)
    kernel_xx = gaussian_kernel(x, x, sigma)
    kernel_yy = gaussian_kernel(y, y, sigma)
    kernel_xy = gaussian_kernel(x, y, sigma)

    mmd = np.sum(kernel_xx) / (n**2) + np.sum(kernel_yy) / (m**2) - 2 * np.sum(kernel_xy) / (n * m)
    return mmd

def calculate_ks_distance(metric, data, true_labels, k=5, alpha = 0.5):
    """Calculate Kolmogorov-Smirnov distance between the local density distributions of two classes."""
    if metric == 'density':
        # Calculate local density for all data points
        local_metric = calculate_local_density(data, k)
    elif metric == 'distance':
        # Calculate local average distance for all data points
        local_metric = calculate_local_distance(data, k)
    elif metric == 'dimension':
        # Calculate local average dimensionality for all data points
        local_metric = mle_lid(data, k)
    elif metric == 'nlabels':
        # Calculate local absolute mean of neighbours labels for all data points
        local_metric = calculate_local_nl(data, true_labels, k)

    # Split local properties based on true class labels
    metric_class_0 = local_metric[true_labels == 0]
    metric_class_1 = local_metric[true_labels == 1]

    mean_local_metric = np.mean(local_metric)
    std_local_metric = np.std(local_metric)

    # Calculate weighted separability score
    overall_separability = alpha * mean_local_metric + (1-alpha) * std_local_metric

    # Calculate the K-S distance between these two groups
    ks_statistic, _ = ks_2samp(metric_class_0, metric_class_1)

    return ks_statistic, overall_separability

def calculate_wasserstein_distance(metric, data, true_labels, k=5):
    if metric == 'density':
        # Calculate local density for all data points
        local_metric = calculate_local_density(data, k)
    elif metric == 'distance':
        # Calculate local average distance for all data points
        local_metric = calculate_local_distance(data, k)
    elif metric == 'dimension':
        # Calculate local average distance for all data points
        local_metric = mle_lid(data, k)
    # Split local metric based on true class labels
    metric_class_0 = local_metric[true_labels == 0]
    metric_class_1 = local_metric[true_labels == 1]

    # Calculate Wasserstein Distance
    w_distance = wasserstein_distance(metric_class_0, metric_class_1)

    return w_distance

def calculate_bhattacharyya_distance(metric, data, true_labels, k=5):
    if metric == 'density':
        # Calculate local density for all data points
        local_metric = calculate_local_density(data, k)
    elif metric == 'distance':
        # Calculate local average distance for all data points
        local_metric = calculate_local_distance(data, k)
    elif metric == 'dimension':
        # Calculate local average distance for all data points
        local_metric = mle_lid(data, k)
    # Split local metric based on true class labels
    metric_class_0 = local_metric[true_labels == 0]
    metric_class_1 = local_metric[true_labels == 1]

    # Calculate means and variances
    mean_0, var_0 = np.mean(metric_class_0), np.var(metric_class_0)
    mean_1, var_1 = np.mean(metric_class_1), np.var(metric_class_1)

    # Calculate Bhattacharyya Distance
    coeff = 1 / 8 * ((mean_0 - mean_1) ** 2) / (var_0 + var_1) / 2 + 0.5 * np.log((var_0 + var_1) / 2 / sqrt(var_0 * var_1))
    b_distance = sqrt(1 - exp(-coeff))

    return b_distance

def calculate_dunns_index(data, labels, normalize=True):
    """Calculate Dunn's Index for the given clustering.

    Parameters:
        data (np.ndarray): Data matrix of shape (n_samples, n_features)
        labels (np.ndarray): Cluster labels for each sample
        normalize (bool): If True, returns value in [0,1] range

    Returns:
        float: Dunn's Index (normalized if specified)
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0  # Not defined for fewer than 2 clusters

    # Compute pairwise distances
    distances = squareform(pdist(data))
    np.fill_diagonal(distances, np.inf)

    # Compute minimum inter-cluster distance
    min_inter_cluster_distance = np.inf
    for i in unique_labels:
        for j in unique_labels:
            if i < j:  # Avoid double computation
                d_ij = np.min(distances[np.ix_(labels == i, labels == j)])
                min_inter_cluster_distance = min(min_inter_cluster_distance, d_ij)

    # Compute maximum intra-cluster distance (cluster diameter)
    max_intra_cluster_distance = 0
    for label in unique_labels:
        cluster_points = data[labels == label]
        if len(cluster_points) > 1:
            intra_distances = pdist(cluster_points)
            max_diameter = np.max(intra_distances)
            max_intra_cluster_distance = max(max_intra_cluster_distance, max_diameter)

    if max_intra_cluster_distance == 0:
        return 0.0  # Avoid division by zero

    # Raw Dunn's Index
    dunn_index = min_inter_cluster_distance / max_intra_cluster_distance

    # Optional normalization
    if normalize:
        return dunn_index / (1.0 + dunn_index)
    else:
        return dunn_index



def calculate_silhouette_score(data, labels, normalize=True):
    """Calculate Silhouette Score for the given clustering.

    Parameters:
        data (np.ndarray): Data matrix of shape (n_samples, n_features)
        labels (np.ndarray): Cluster labels for each sample
        normalize (bool): If True, map to [0,1] range

    Returns:
        float: Silhouette Score (normalized if specified)
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0  # Not defined for fewer than 2 clusters

    raw_score = silhouette_score(data, labels)

    if normalize:
        return (raw_score + 1.0) / 2.0  # Map from [-1, 1] to [0, 1]
    else:
        return raw_score

def calculate_davies_bouldin_index(data, labels, normalize=True):
    """Calculate Davies–Bouldin Index for the given clustering.

    Parameters:
        data (np.ndarray): Data matrix of shape (n_samples, n_features)
        labels (np.ndarray): Cluster labels for each sample
        normalize (bool): If True, map to [0,1] using inverse saturation

    Returns:
        float: Davies–Bouldin Index (normalized if specified)
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0  # Not defined for fewer than 2 clusters

    dbi = davies_bouldin_score(data, labels)

    if normalize:
        return 1.0 / (1.0 + dbi)  # Inverse saturation normalization
    else:
        return dbi

def calculate_N2(data, labels):
    data = np.array(data)
    labels = np.array(labels)
    dist_matrix = distance_matrix(data, data)

    total_intra_class_dist = 0
    total_inter_class_dist = 0

    for i, point in enumerate(data):
        same_class = labels == labels[i]
        different_class = labels != labels[i]

        # Handling intra-class distances
        same_class_distances = dist_matrix[i, same_class]
        same_class_distances_mod = np.where(same_class_distances == 0, np.inf, same_class_distances)
        total_intra_class_dist += np.min(same_class_distances_mod)

        # Handling inter-class distances
        different_class_distances = dist_matrix[i, different_class]
        total_inter_class_dist += np.min(different_class_distances)

    # Calculating the intra_extra ratio then N2 score based on the paper's formula
    intra_extra_ratio = total_intra_class_dist / total_inter_class_dist
    N2_score = intra_extra_ratio / (1 + intra_extra_ratio)

    return 1-N2_score

def generate_interpolated_data(X, y):
    interpolated_data = []
    parent_indices = []  # Keep track of the parent indices for each interpolated point
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if y[i] == y[j]:
                new_point = (X[i] + X[j]) / 2
                interpolated_data.append(new_point)
                parent_indices.append(i)  # Store the index of one of the parents
    return np.array(interpolated_data), parent_indices

def calculate_N4(data, labels):
    interpolated_X, parent_indices = generate_interpolated_data(data, labels)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(data, labels)
    interpolated_y = knn.predict(interpolated_X)
    parent_y = [labels[idx] for idx in parent_indices]  # Use the correct parent index for each interpolated point

    error_rate = np.mean(interpolated_y != parent_y)
    return 1-error_rate

def calculate_radii(data, labels):
    distances = squareform(pdist(data))
    radii = np.zeros(len(data))

    for i in range(len(data)):
        different_class = labels != labels[i]
        nearest_enemy_dist = np.min(distances[i, different_class])
        radii[i] = nearest_enemy_dist / 2

    return radii, distances

def check_overlaps(radii, distances):
    n = len(radii)
    overlaps = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] < (radii[i] + radii[j]):
                if radii[i] < radii[j]:
                    overlaps[i] = True
                else:
                    overlaps[j] = True

    return overlaps

def calculate_T1(data, labels):
    radii, distances = calculate_radii(data, labels)
    overlaps = check_overlaps(radii, distances)
    # Count only unique hyperspheres (not overlapped)
    unique_hyperspheres = np.sum(~overlaps)
    t1_value = unique_hyperspheres / len(data)
    return 1 - t1_value

def nearest_enemy_distances(data, labels):
    distances = squareform(pdist(data))
    nearest_enemy_dist = np.zeros(distances.shape[0])

    for i in range(distances.shape[0]):
        # Filter out same class distances by setting them to a high value
        same_class = labels[i] == labels
        distances[i, same_class] = np.max(distances) + 1
        nearest_enemy_dist[i] = np.min(distances[i])

    return nearest_enemy_dist

def calculate_LSC(data, labels):
    distances = squareform(pdist(data))
    nearest_enemy_dist = nearest_enemy_distances(data, labels)
    local_sets = [np.sum(distances[i] < nearest_enemy_dist[i]) for i in range(len(data))]

    LSC = np.sum(local_sets) / (len(data) ** 2)
    return LSC

