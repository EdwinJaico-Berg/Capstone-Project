import numpy as np
import pandas as pd

def GetBestLatLon(coordinates: np.array, station_coordinates: np.array) -> tuple:
    """
    Takes in a latitude or longitude value and returns the coordinates of the
    closest node  

    Parameters
    ----------
    coordinates: array of the coordinate we are trying to find the closest node to
    station_coordinates: array of all coordinates of the emissions data

    Returns
    -------
    A tuple containing the best latitude and longitude values
    """
    difference = coordinates - station_coordinates
    lat = station_coordinates[np.argmin(abs(difference[:, 0]))][0]
    lon = station_coordinates[np.argmin(abs(difference[:, 1]))][1]
    
    return lat, lon

def GetEmissions(df: pd.DataFrame, df2: pd.DataFrame) -> dict:
    '''
    Function that takes in the wildfire DataFrame, extracting the latitude and longitude
    values and finding the emissions data from the gridded node that is closest.

    Parameters
    ----------
    df: wildfire DataFrame with wildfire data
    emissions_df: DataFrame of the emissions data

    Returns
    -------
    Dictionary containing the emissions data and the coordinates from which the data was
    collected
    '''
    emissions_df = df2.copy()

    # Convert longitude into correct range
    emissions_df['lon'] = emissions_df['lon'].apply(lambda x: x - 360)

    # Get np.array of relevant columns
    emissions_data = np.array(emissions_df)

    # Get the latitude and longitude values
    # We will get them for a specific year to avoid repetitions
    emissions_coords = \
        np.array(emissions_df[emissions_df['year'] == 1992][['lat', 'lon']])

    data = {
        'emission': [],
        'coordinates': []
    }

    for _, row in df.iterrows():
        
        # Get the coordinates of the row
        wildfire_coords = np.array([row['LATITUDE'], row['LONGITUDE']])
        
        # Get the best latitude and longitude values
        lat, lon = GetBestLatLon(wildfire_coords, emissions_coords)
        
        data['coordinates'].append((lat, lon))

        index = np.where((emissions_data[:,0] == lat) & (emissions_data[:,1] == lon) \
            & (emissions_data[:,-1] == row['FIRE_YEAR']))

        data['emission'].append(emissions_data[index][0][2])
    
    return data