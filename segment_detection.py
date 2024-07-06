import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

def is_stay_location(df_batch):
    """
    Determine if there are stay locations in the given batch of trajectory data using DBSCAN clustering.

    Args:
        df_batch (pd.DataFrame): DataFrame containing 'lon' (longitude) and 'lat' (latitude) columns.

    Returns:
        bool: True if the number of unique clusters is less than or equal to 10, indicating stay locations. False otherwise.
    """
    coordinates = df_batch[['lon', 'lat']].values
    dbscan = DBSCAN(eps=1e-6, min_samples=1)
    labels = dbscan.fit_predict(coordinates)

    if len(np.unique(labels)) <= 10:
        return True
    return False

def calculate_curvature(x, y):
    """
    Calculate the curvature of each point in the trajectory.

    Args:
        x (np.array): Array of x coordinates (longitude).
        y (np.array): Array of y coordinates (latitude).

    Returns:
        np.array: Array of curvature values for each point.
    """
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt**2 + dy_dt**2)**(3/2)
    return curvature

def is_turning(df_batch):
    """
    Determine if there are sharp turns in the given batch of trajectory data.

    Args:
        df_batch (pd.DataFrame): DataFrame containing 'lon' (longitude) and 'lat' (latitude) columns.

    Returns:
        bool: True if there is a sharp turn (curvature > 1e5), False otherwise.
    """
    curvatures = calculate_curvature(df_batch['lon'].values, df_batch['lat'].values)
    if np.max(curvatures) > 1e5:
        return True
    return False
