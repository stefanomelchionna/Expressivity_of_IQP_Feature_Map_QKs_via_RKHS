'''
Synthetic dataset generators for the numerical experiments of Section IV of the paper:

- periodic data (Section IV A, Eq. (6)):    compute_periodic_data_classification, create_periodic_dataset_AD
- multimodal data (Section IV B, Eq. (7)):  create_blob_multimodal_classification, create_blob_with_outliers
'''


import pandas as pd
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler


######################################################
#       CLASSIFICATION
######################################################

################## PERIODIC DATASET GENERATION ##################

def input_function(x_train, oscillations=5, number_of_features=2):
    """
    Labels of the periodic dataset, Eq. (6) of the paper: y(x) = (sign(sum_j sin(2 pi omega x_j)) + 1) / 2.
    Parameters:
        x_train: Input data, shape (n_samples, number_of_features).
        oscillations: Number of oscillations omega of the sine function in [0, 1] (the frequency).
        number_of_features: Number of features in the input data.
    Returns:
        y: Labels in {0, 1}.
    """
    y = []
    for i in range(x_train.shape[0]):
        p = 0
        for j in range(0, number_of_features):
            p = p + np.sin(x_train[i][j] * 2*np.pi * oscillations)
        y.append(np.sign(p))
    y = 0.5 * (np.array(y) + 1)
    return y


def compute_periodic_data_classification(input_function=input_function, oscillations=5, sample_size=50, seed=10, number_of_features=2):
    """
    Generates data uniformly at random in [0, 1]^D and labels them with input_function.
    Parameters:
        input_function: Function assigning the labels.
        oscillations: Number of oscillations omega of the labelling function.
        sample_size: Number of samples to generate.
        seed: Seed for random number generation.
        number_of_features: Number of features D.
    Returns:
        x_train: Generated input data, shape (sample_size, number_of_features).
        y_train: Labels in {0, 1}.
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Generate random input data
    x_train = []
    for i in range(number_of_features):
        x_train.append(np.random.uniform(0, 1, size=sample_size))

    x_train = np.array(x_train).transpose()

    # Assign the labels
    y_train = input_function(np.array(x_train), oscillations, number_of_features)

    return x_train, y_train

################ MULTIMODAL DATASET GENERATION ##################

def create_blob_multimodal_classification(n_features = 4, random_state=42, number_of_blobs = 4, sample_size=120):
    """
    Two multimodal classes, each a mixture of `number_of_blobs` Gaussian blobs with different, randomly chosen centres
    (make_blobs with seeds random_state and random_state + 10). Balanced dataset.
    Returns:
        df: DataFrame with columns Feature_1..Feature_D and label (0/1).
    """
    class_0=make_blobs(n_samples=int(sample_size/2), n_features=n_features, centers=number_of_blobs, random_state=random_state)
    class_1=make_blobs(n_samples=int(sample_size/2), n_features=n_features, centers=number_of_blobs, random_state=random_state+10)

    #merge and add labels
    X_train = np.vstack([class_0[0], class_1[0]])
    y_train = np.hstack([np.zeros(len(class_0[0])), np.ones(len(class_1[0]))])
    # Convert to DataFrame
    df = pd.DataFrame(X_train, columns=[f'Feature_{i+1}' for i in range(n_features)])
    df['label'] = y_train
    return df


#######################################################################
#    ANOMALY DETECTION
#######################################################################

################PERIODIC DATASET GENERATION WITH OUTLIERS##################

def create_periodic_dataset_AD(oscillations=5, sample_size=100, ratio_outliers=0.2, number_of_features=2, random_state=42):
    """
    Periodic anomaly-detection dataset: the inliers are the points labelled 0 by Eq. (6) out of 2 * sample_size
    uniform points; round(sample_size * ratio_outliers) outliers are drawn uniformly in [0, 1]^D.
    Returns:
        df: DataFrame with columns Feature_1..Feature_D and label (0 = inlier, 1 = outlier).
    """
    X_train,y_train = compute_periodic_data_classification(oscillations=oscillations, sample_size=sample_size*2, number_of_features=number_of_features, seed=random_state)
    #filter class 0 only and then add outliers
    X_train = X_train[y_train == 0]
    y_train = y_train[y_train == 0]

    minority_samples = round(sample_size * ratio_outliers)
    minority_class = np.random.uniform(low=0, high=1, size=(minority_samples, number_of_features))
    # Combine the majority class and the minority class (outliers)
    data = np.vstack([X_train, minority_class])
    labels = np.hstack([np.zeros(len(X_train)), np.ones(minority_samples)])  # Majority class is labeled 0, minority class (outliers) is labeled 1
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[f'Feature_{i+1}' for i in range(number_of_features)])
    df['label'] = labels
    return df


################MULTIMODAL DATASET GENERATION WITH OUTLIERS##################

def create_blob_with_outliers(n_features = 4, random_state=42, number_of_blobs = 4, sample_size=120, ratio_outliers = 0.2):
    """
    Generates a dataset with a specified number of features. The dataset is composed by blobs and outliers. Blobs are used to represent a multimodal distribution. Other points are generated at random in the range [-10, 10]. The data are then min-max scaled to [0, 1].
    Parameters:
        n_features: Number of features for each sample (default is 4).
        random_state: Seed for random number generation to ensure reproducibility (default is 42).
        number_of_blobs: Number of distinct blobs (clusters) to generate (default is 4).
        sample_size: Total number of samples to generate (default is 120).
        ratio_outliers: Proportion of outliers to normal points in the dataset (default is 0.2).
    Returns:
        df: Pandas DataFrame with columns Feature_1..Feature_D and label (0 = inlier, 1 = outlier).
    """

    #compute the number of samples for the majority and the minority class
    majority_samples = round(sample_size/(1+ratio_outliers))
    minority_samples = sample_size - majority_samples

    #fix the random state for replicability
    np.random.seed(random_state)

    # Create majority class blob
    # Note that the default option center_box=(-10.0, 10.0) implies that centers are chosen at random in the range [-10, 10], also the default cluster_std = 1.0 means that the standard deviation is 1.0 for each blob
    majority_blob, _ = make_blobs(n_samples=majority_samples, n_features=n_features, centers=number_of_blobs, random_state=random_state)

    # Create minority class outliers uniformly at random in the range [-10, 10]
    minority_blob = np.random.uniform(low=-10, high=10, size=(minority_samples, n_features))

    # Combine the blobs and the outliers
    data = np.vstack([majority_blob, minority_blob])
    labels = np.hstack([np.zeros(majority_samples), np.ones(minority_samples)]) #majority class is labeled 0 and minority as 1

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[f'Feature_{i+1}' for i in range(n_features)])
    df['label'] = labels

    # Scale the data
    scaler = MinMaxScaler()
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

    return df
