import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from typing import Tuple, Optional

class LocationCategorizer:
    """
    A class to categorize geographic locations into quadrants for Kenya fire data.
    """
    
    def __init__(self, use_kenya_center=True):
        """
        Initialize the categorizer.
        Parameters:
        use_kenya_center (bool): If True, uses Kenya's real geographic center. 
                                If False, calculates center from the data.
        """
        self.use_kenya_center = use_kenya_center
        self.kenya_center_lat = None
        self.kenya_center_lon = None
        
        if use_kenya_center:
            self.kenya_center_lat, self.kenya_center_lon = self.get_kenya_center()
            print(f"Kenya's real geographic center: ({self.kenya_center_lat:.4f}, {self.kenya_center_lon:.4f})")
    
    def get_kenya_center(self) -> Tuple[float, float]:
        """
        Get Kenya's real geographic center using Nominatim API (OpenStreetMap).
        
        Returns:
        tuple: (center_latitude, center_longitude)
        """
        # Use Nominatim API (OpenStreetMap)
        try:
            center_lat, center_lon = self._get_center_from_nominatim()
            if center_lat and center_lon:
                return center_lat, center_lon
        except Exception as e:
            print(f"Nominatim API failed: {e}")
            raise Exception("Could not determine Kenya's center from Nominatim API")
    
    def _get_center_from_nominatim(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get Kenya's center using Nominatim API.
        """
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': 'Kenya',
            'format': 'json',
            'limit': 1,
            'polygon_geojson': 1
        }
        
        headers = {
            'User-Agent': 'FireLocationCategorizer/1.0'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data and len(data) > 0:
            # Get bounding box
            bbox = data[0].get('boundingbox')
            if bbox:
                south, north, west, east = map(float, bbox)
                center_lat = (north + south) / 2
                center_lon = (east + west) / 2
                print(f"Found Kenya center via Nominatim: ({center_lat:.4f}, {center_lon:.4f})")
                return center_lat, center_lon
        
        return None, None
    
    def get_kenya_boundaries(self) -> dict:
        """
        Get Kenya's geographic boundaries using Nominatim API.
        
        Returns:
        dict: Dictionary with 'north', 'south', 'east', 'west' boundaries
        """
        # Get boundaries from Nominatim API
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': 'Kenya',
                'format': 'json',
                'limit': 1
            }
            headers = {'User-Agent': 'FireLocationCategorizer/1.0'}
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data and len(data) > 0:
                bbox = data[0].get('boundingbox')
                if bbox:
                    south, north, west, east = map(float, bbox)
                    return {
                        'north': north,
                        'south': south,
                        'east': east,
                        'west': west
                    }
        except Exception as e:
            print(f"Could not get Kenya boundaries from Nominatim API: {e}")
            return None
    
    def calculate_center_from_data(self, df):
        """
        Calculate the center point from the data itself.
        
        Parameters:
        df (pandas.DataFrame): DataFrame with 'latitude' and 'longitude' columns
        
        Returns:
        tuple: (center_latitude, center_longitude)
        """
        center_lat = (df['latitude'].min() + df['latitude'].max()) / 2
        center_lon = (df['longitude'].min() + df['longitude'].max()) / 2
        return center_lat, center_lon
    
    def categorize_location(self, latitude, longitude, center_lat, center_lon):
        """
        Categorize a single location into a quadrant.
        
        Parameters:
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        center_lat (float): Center latitude reference point
        center_lon (float): Center longitude reference point
        
        Returns:
        str: Quadrant category ('Northeast', 'Northwest', 'Southeast', 'Southwest')
        """
        if latitude >= center_lat and longitude >= center_lon:
            return 'Northeast'
        elif latitude >= center_lat and longitude < center_lon:
            return 'Northwest'
        elif latitude < center_lat and longitude >= center_lon:
            return 'Southeast'
        else:  # latitude < center_lat and longitude < center_lon
            return 'Southwest'
    
    def process_fire_data(self, input_file, output_file=None):
        """
        Process fire data and add quadrant categorization.
        
        Parameters:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
        
        Returns:
        pandas.DataFrame: DataFrame with added 'quadrant' column
        """
        # Read the data
        df = pd.read_csv(input_file)
        
        # Check if required columns exist
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            raise ValueError("Dataset must contain 'latitude' and 'longitude' columns")
        
        # Determine center point
        if self.use_kenya_center:
            center_lat, center_lon = self.kenya_center_lat, self.kenya_center_lon
            print(f"Using Kenya's real geographic center: ({center_lat:.4f}, {center_lon:.4f})")
        else:
            center_lat, center_lon = self.calculate_center_from_data(df)
            print(f"Using data center: ({center_lat:.4f}, {center_lon:.4f})")
        
        # Apply categorization
        df['quadrant'] = df.apply(
            lambda row: self.categorize_location(
                row['latitude'], row['longitude'], center_lat, center_lon
            ), axis=1
        )
        
        # Print summary statistics
        print("\n=== Quadrant Distribution ===")
        quadrant_counts = df['quadrant'].value_counts()
        for quadrant, count in quadrant_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{quadrant}: {count} fires ({percentage:.1f}%)")
        
        print(f"\n=== Coordinate Ranges ===")
        print(f"Latitude: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
        print(f"Longitude: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
        
        # Show Kenya boundaries for reference
        boundaries = self.get_kenya_boundaries()
        print(f"\n=== Kenya Boundaries (for reference) ===")
        print(f"North: {boundaries['north']:.4f}°, South: {boundaries['south']:.4f}°")
        print(f"East: {boundaries['east']:.4f}°, West: {boundaries['west']:.4f}°")
        
        # Save to file if output path provided
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\nCategorized data saved to: {output_file}")
        
        return df
    
    def visualize_quadrants(self, df, save_plot=False, plot_filename='quadrant_map.png'):
        """
        Create a visualization of the fire locations by quadrant.
        
        Parameters:
        df (pandas.DataFrame): DataFrame with categorized data
        save_plot (bool): Whether to save the plot
        plot_filename (str): Filename for saved plot
        """
        plt.figure(figsize=(14, 10))
        
        # Create scatter plot with different colors for each quadrant
        colors = {'Northeast': 'red', 'Northwest': 'blue', 
                 'Southeast': 'green', 'Southwest': 'orange'}
        
        for quadrant in df['quadrant'].unique():
            subset = df[df['quadrant'] == quadrant]
            plt.scatter(subset['longitude'], subset['latitude'], 
                       c=colors[quadrant], label=quadrant, alpha=0.7, s=50)
        
        # Add center point
        if self.use_kenya_center:
            center_lat, center_lon = self.kenya_center_lat, self.kenya_center_lon
            center_label = "Kenya's Geographic Center"
        else:
            center_lat, center_lon = self.calculate_center_from_data(df)
            center_label = "Data Center"
            
        plt.scatter(center_lon, center_lat, c='black', marker='x', s=200, 
                   linewidth=3, label=center_label)
        
        # Add grid lines at center
        plt.axhline(y=center_lat, color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=center_lon, color='black', linestyle='--', alpha=0.5)
        
        # Add Kenya boundaries
        boundaries = self.get_kenya_boundaries()
        plt.axhline(y=boundaries['north'], color='gray', linestyle=':', alpha=0.7, label='Kenya Boundaries')
        plt.axhline(y=boundaries['south'], color='gray', linestyle=':', alpha=0.7)
        plt.axvline(x=boundaries['east'], color='gray', linestyle=':', alpha=0.7)
        plt.axvline(x=boundaries['west'], color='gray', linestyle=':', alpha=0.7)
        
        plt.xlabel('Longitude (°E)')
        plt.ylabel('Latitude (°N)')
        plt.title('Fire Locations by Quadrant in Kenya\n(Using Real Geographic Center)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {plot_filename}")
        
        plt.tight_layout()
        plt.show()


def process_aggregated_fire_data(input_file='fire_data_2000-18.csv', output_file='fire_data_2000-18_categorized.csv'):
    """
    Process the aggregated fire data and add quadrant categorization with one-hot encoding.
    
    Parameters:
    input_file (str): Path to input CSV file (default: 'fire_data_2000-18.csv')
    output_file (str): Path to output CSV file (default: 'fire_data_2000-18_categorized.csv')
    
    Returns:
    pandas.DataFrame: DataFrame with added quadrant columns
    """
    print(f"Processing aggregated fire data from: {input_file}")
    
    # Initialize categorizer with Kenya's real center
    categorizer = LocationCategorizer(use_kenya_center=True)
    
    # Read the aggregated data
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} records from {input_file}")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found. Please run 2_data_aggregate.py first.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    # Check if required columns exist
    if 'mean_latitude' not in df.columns or 'mean_longitude' not in df.columns:
        print("Error: Dataset must contain 'mean_latitude' and 'mean_longitude' columns")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Get Kenya's center coordinates
    center_lat, center_lon = categorizer.kenya_center_lat, categorizer.kenya_center_lon
    print(f"Using Kenya's real geographic center: ({center_lat:.4f}, {center_lon:.4f})")
    
    # Apply categorization using mean_latitude and mean_longitude
    df['quadrant'] = df.apply(
        lambda row: categorizer.categorize_location(
            row['mean_latitude'], row['mean_longitude'], center_lat, center_lon
        ), axis=1
    )
    
    # Create one-hot encoded columns for quadrants
    print("Creating one-hot encoded columns for quadrants...")
    df['quadrant_NE'] = (df['quadrant'] == 'Northeast').astype(int)
    df['quadrant_NW'] = (df['quadrant'] == 'Northwest').astype(int) 
    df['quadrant_SE'] = (df['quadrant'] == 'Southeast').astype(int)
    df['quadrant_SW'] = (df['quadrant'] == 'Southwest').astype(int)
    
    # Print summary statistics
    print("\n=== Quadrant Distribution ===")
    quadrant_counts = df['quadrant'].value_counts()
    for quadrant, count in quadrant_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{quadrant}: {count} months ({percentage:.1f}%)")
    
    print(f"\n=== Coordinate Ranges ===")
    print(f"Mean Latitude: {df['mean_latitude'].min():.4f} to {df['mean_latitude'].max():.4f}")
    print(f"Mean Longitude: {df['mean_longitude'].min():.4f} to {df['mean_longitude'].max():.4f}")
    
    # Show Kenya boundaries for reference
    boundaries = categorizer.get_kenya_boundaries()
    if boundaries:
        print(f"\n=== Kenya Boundaries (for reference) ===")
        print(f"North: {boundaries['north']:.4f}°, South: {boundaries['south']:.4f}°")
        print(f"East: {boundaries['east']:.4f}°, West: {boundaries['west']:.4f}°")
    
    # Verify one-hot encoding
    print(f"\n=== One-Hot Encoding Verification ===")
    print(f"Northeast (NE): {df['quadrant_NE'].sum()} months")
    print(f"Northwest (NW): {df['quadrant_NW'].sum()} months") 
    print(f"Southeast (SE): {df['quadrant_SE'].sum()} months")
    print(f"Southwest (SW): {df['quadrant_SW'].sum()} months")
    print(f"Total check: {df['quadrant_NE'].sum() + df['quadrant_NW'].sum() + df['quadrant_SE'].sum() + df['quadrant_SW'].sum()} (should equal {len(df)})")
    
    # Save to file
    try:
        df.to_csv(output_file, index=False)
        print(f"\nCategorized data with one-hot encoding saved to: {output_file}")
        
        # Show sample of new columns
        print(f"\n=== Sample of categorized data ===")
        sample_cols = ['month', 'year', 'mean_latitude', 'mean_longitude', 'quadrant', 
                      'quadrant_NE', 'quadrant_NW', 'quadrant_SE', 'quadrant_SW']
        print(df[sample_cols].head(10))
        
    except Exception as e:
        print(f"Error saving file: {e}")
        return None
    
    return df


def main():
    """
    Main function to process the aggregated fire data with quadrant categorization.
    """
    print("=== Fire Data Quadrant Categorization ===")
    print("Processing aggregated fire data (2000-2018) with one-hot encoded quadrants")
    
    # Process the aggregated fire data
    df_categorized = process_aggregated_fire_data()
    
    if df_categorized is not None:
        print(f"\n=== Final Dataset Info ===")
        print(f"Shape: {df_categorized.shape}")
        print(f"Columns: {list(df_categorized.columns)}")
        
        # Create visualization using mean coordinates
        try:
            categorizer = LocationCategorizer(use_kenya_center=True)
            
            # Rename columns temporarily for visualization
            df_viz = df_categorized.copy()
            df_viz['latitude'] = df_viz['mean_latitude']
            df_viz['longitude'] = df_viz['mean_longitude']
            
            categorizer.visualize_quadrants(df_viz, save_plot=True, 
                                          plot_filename='fire_data_quadrants_map.png')
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    else:
        print("Failed to process the fire data.")


# Function for easy batch processing
def categorize_multiple_files(file_paths, use_kenya_center=True):
    """
    Categorize multiple fire data files.
    
    Parameters:
    file_paths (list): List of input file paths
    use_kenya_center (bool): Whether to use Kenya's center or calculate from data
    
    Returns:
    dict: Dictionary with file paths as keys and categorized DataFrames as values
    """
    categorizer = LocationCategorizer(use_kenya_center=use_kenya_center)
    results = {}
    
    for file_path in file_paths:
        try:
            print(f"\nProcessing: {file_path}")
            output_path = file_path.replace('.csv', '_categorized.csv')
            df = categorizer.process_fire_data(file_path, output_path)
            results[file_path] = df
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return results


if __name__ == "__main__":
    main() 