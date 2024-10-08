import webbrowser
import pandas as pd
import exifread
import folium
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
from math import radians, cos, sin, asin, sqrt, pi
import numpy as np
import re

# Load the list of chain restaurants from a CSV file
chain_restaurants_df = pd.read_csv("data/list_of_chain_restaurants.csv")
# Assuming the file has a single column with restaurant names
chain_restaurants_set = set(chain_restaurants_df.iloc[:, 0].str.strip().unique())

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in meters between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    m = 6371000 * c # convert km to meters
    return m

def encode_room_type(data):
    if 'room_type' in data.columns:
        encoder = LabelEncoder()
        data['room_type_encoded'] = encoder.fit_transform(data['room_type'])
        return data, encoder
    else:
        print("'room_type' column not found.")
        return data, None
    
def get_coordinates(photo):
    with open(photo, 'rb') as f:
        exif_dict = exifread.process_file(f)
        lon_ref = exif_dict["GPS GPSLongitudeRef"].printable
        lon = exif_dict["GPS GPSLongitude"].printable[1:-1].replace(" ", "").replace("/", ",").split(",")
        lon = float(lon[0]) + float(lon[1]) / 60 + float(lon[2]) / float(lon[3]) / 3600
        if lon_ref != "E":
            lon = lon * (-1)
        lat_ref = exif_dict["GPS GPSLatitudeRef"].printable
        lat = exif_dict["GPS GPSLatitude"].printable[1:-1].replace(" ", "").replace("/", ",").split(",")
        lat = float(lat[0]) + float(lat[1]) / 60 + float(lat[2]) / float(lat[3]) / 3600
        if lat_ref != "N":
            lat = lat * (-1)
    return lat, lon

def distance(hotel_range):
    p = pi / 180
    lat1 = hotel_range['lat']
    lon1 = hotel_range['lon']
    lat2 = hotel_range['latitude']
    lon2 = hotel_range['longitude']
    a = 0.5 - np.cos((lat2 - lat1) * p) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    b = 12742 * np.arcsin(np.sqrt(a)) * 1000
    return b

def has_tags(tags):
    return 'tourism' in tags

def extract_rating(name):
    try:
        rating = re.search(r'★(\d+\.\d+)', name).group(1)
        return float(rating)
    except AttributeError:
        return None

# Load the listings.csv file
data = pd.read_csv("data/listings.csv")

# Apply the extract_rating function to create a new 'rating' column
data['rating'] = data['name'].apply(extract_rating)

# Optional: Fill missing ratings with a default value, e.g., the mean rating
mean_rating = data['rating'].dropna().mean()
data['rating'] = data['rating'].fillna(mean_rating)

# Overwrite the listings.csv file with the updated DataFrame
data.to_csv("data/listings.csv", index=False)

def getdata(frame):
    temp = frame[frame['tags'].apply(has_tags)]
    temp_of_four = frame[(frame['amenity'] == 'food_court') |
                                (frame['amenity'] == 'atm') |
                                (frame['amenity'] == 'bus_station') |
                                (frame['amenity'] == 'clinic')|
                                (frame['amenity'] == 'fast_food')| 
                                (frame['amenity'] == 'marketplace')]
    frames = [temp, temp_of_four]
    result = pd.concat(frames)
    return result

def categorize_restaurants(frame):
    # Ensure working with a copy to avoid SettingWithCopyWarning
    frame = frame.copy()
    
    # Proceed with the existing logic...
    frame['name'] = frame['name'].fillna('Unknown Restaurant')
    name_counts = frame['name'].value_counts()
    frame['is_chain'] = frame['name'].apply(lambda x: name_counts[x] > 1)
    filtered_frame = frame[(frame['amenity'] == 'fast_food') | (frame['amenity'] == 'restaurant') | (frame['amenity'] == 'food_court')]
    
    return filtered_frame



def display_chain_non_chain_restaurants(restaurants_df, lat, lon, search_range, amenities_data):
    restaurants_map = folium.Map(location=[lat, lon], zoom_start=16)

    # Apply distance calculation and filter here, as in the previous adjustment
    restaurants_df['distance'] = restaurants_df.apply(
        lambda row: haversine(lon, lat, row['lon'], row['lat']), axis=1)
    within_range_restaurants = restaurants_df[restaurants_df['distance'] <= search_range]
    chain_restaurants = restaurants_df[restaurants_df['is_chain'] == True]
    for idx, row in chain_restaurants.iterrows():
        folium.Marker(
            [row['lat'], row['lon']],  # Adjusted to match your column names
            icon=folium.Icon(color='blue', icon='cutlery'),
            popup=f"Chain: {row['name']}",
        ).add_to(restaurants_map)

    # Add markers for non-chain restaurants
    non_chain_restaurants = restaurants_df[restaurants_df['is_chain'] == False]
    for idx, row in non_chain_restaurants.iterrows():
        folium.Marker(
            [row['lat'], row['lon']],  # Adjusted to match your column names
            icon=folium.Icon(color='green', icon='cutlery'),
            popup=f"Non-Chain: {row['name']}",
        ).add_to(restaurants_map)

    # Extract amenities data for heatmap
    amenities_heatmap_data = getdata(amenities_data)[['lat', 'lon']].values.tolist()

    # Add HeatMap for amenities
    HeatMap(amenities_heatmap_data, radius=15).add_to(restaurants_map)

def add_user_location(map_, lat, lon):
    folium.Marker(
        [lat, lon], 
        icon=folium.Icon(color='red', icon='info-sign'), 
        popup='Your Location'
    ).add_to(map_)





def prepare_and_display_airbnb_listings(data, lat, lon, max_budget, search_range, room_type, min_rating, amenities_data, recommended_id=None):
    airbnb_map = folium.Map(location=[lat, lon], zoom_start=16)

    # Filter data based on max budget, min rating, and calculate distance
    data['distance'] = data.apply(lambda x: haversine(lon, lat, x['longitude'], x['latitude']), axis=1)
    filtered_listings = data[(data['distance'] <= search_range) & (data['rating'] >= min_rating)]
    if max_budget.lower() != "all":
        filtered_listings = filtered_listings[filtered_listings['price'] <= float(max_budget)]

    # Check if room_type filter is applied
    if room_type != "all":
        room_type_map = {"1": "entire home/apt", "2": "private room"}
        selected_room_type = room_type_map.get(room_type, "Unknown")
        filtered_listings = filtered_listings[filtered_listings['room_type'] == selected_room_type]

    #print(f"Number of filtered listings: {len(filtered_listings)}")

    # Loop through filtered Airbnb listings and add markers
    for idx, listing in filtered_listings.iterrows():
        if recommended_id is not None and listing['id'] == recommended_id:
            folium.Marker(
                [listing['latitude'], listing['longitude']], 
                icon=folium.Icon(color='blue', icon='home'),
                popup=f"{listing['name']} - Price: ${listing['price']} - Rating: {listing['rating']}"
            ).add_to(airbnb_map)
        else:
            folium.Marker(
                [listing['latitude'], listing['longitude']], 
                icon=folium.Icon(color='green', icon='home'),
                popup=f"{listing['name']} - Price: ${listing['price']} - Rating: {listing['rating']}"
            ).add_to(airbnb_map)

    # Add user's location
    folium.Marker(
        [lat, lon], 
        icon=folium.Icon(color='red', icon='info-sign'), 
        popup='Your Location'
    ).add_to(airbnb_map)

    # Extract amenities data for heatmap
    amenities_heatmap_data = getdata(amenities_data)[['lat', 'lon']].values.tolist()

    # Add HeatMap for amenities
    HeatMap(amenities_heatmap_data, radius=15).add_to(airbnb_map)

    # Save and open the map
    airbnb_map.save('results/airbnb_listings_map.html')
    webbrowser.open('results/airbnb_listings_map.html', new=2)


def display_chain_non_chain_restaurants(restaurants_df, lat, lon, search_range, amenities_data):
    # Generate the map centered on the given coordinates
    restaurants_map = folium.Map(location=[lat, lon], zoom_start=16)

    # Calculate the distance of each restaurant from the user's location
    restaurants_df['distance'] = restaurants_df.apply(
        lambda row: haversine(lon, lat, row['lon'], row['lat']), axis=1)

    # Filter the DataFrame to only include restaurants within the search range
    within_range_restaurants = restaurants_df[restaurants_df['distance'] <= search_range]

    # Split the filtered restaurants into chain and non-chain
    chain_restaurants = within_range_restaurants[within_range_restaurants['is_chain']]
    non_chain_restaurants = within_range_restaurants[~within_range_restaurants['is_chain']]

    # Add user's location
    folium.Marker(
        [lat, lon], 
        icon=folium.Icon(color='red', icon='info-sign'), 
        popup='Your Location'
    ).add_to(restaurants_map)

    # Add chain restaurants to the map
    for _, row in chain_restaurants.iterrows():
        folium.Marker(
            [row['lat'], row['lon']], 
            icon=folium.Icon(color='blue', icon='cutlery'),
            popup=f"Chain: {row['name']}",
        ).add_to(restaurants_map)

    # Add non-chain restaurants to the map
    for _, row in non_chain_restaurants.iterrows():
        folium.Marker(
            [row['lat'], row['lon']], 
            icon=folium.Icon(color='green', icon='cutlery'),
            popup=f"Non-Chain: {row['name']}",
        ).add_to(restaurants_map)

    # Extract amenities data for heatmap
    amenities_heatmap_data = getdata(amenities_data)[['lat', 'lon']].values.tolist()

    # Add HeatMap for amenities
    HeatMap(amenities_heatmap_data, radius=15).add_to(restaurants_map)
    # Save and open the map
    restaurants_map.save('results/restaurants_map.html')
    webbrowser.open('results/restaurants_map.html', new=2)

def cluster_listings(data):
    # Ensure necessary columns are present for clustering
    required_columns = ['price', 'latitude', 'longitude', 'rating']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"Missing columns for clustering: {missing_columns}")
        return data, None

    features = data[required_columns].copy()
    features.fillna(0, inplace=True)  # Handle NaN values
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    data['cluster'] = kmeans.fit_predict(features_scaled)
    
    return data, kmeans

def recommend_listing(data, user_lat, user_lon, search_range, user_preferences):
    # Filter listings by user's search range
    data['distance'] = data.apply(lambda x: haversine(user_lon, user_lat, x['longitude'], x['latitude']), axis=1)
    within_range_data = data[data['distance'] <= search_range]

    if within_range_data.empty:
        print("No listings within search range.")
        return None, 0, None  # Adding None for the ID of the recommended listing

    # Sort clusters by their average rating
    cluster_ratings = within_range_data.groupby('cluster')['rating'].mean().sort_values(ascending=False)

    total_filtered_listings = 0  # Keep track of total listings that match user preferences

    # Iterate through clusters from highest to lowest average rating
    for best_cluster in cluster_ratings.index:
        cluster_listings = within_range_data[within_range_data['cluster'] == best_cluster]

        if 'max_budget' in user_preferences and user_preferences['max_budget'] != 'all':
            cluster_listings = cluster_listings[cluster_listings['price'] <= user_preferences['max_budget']]
        if 'min_rating' in user_preferences:
            cluster_listings = cluster_listings[cluster_listings['rating'] >= user_preferences['min_rating']]

        total_filtered_listings += len(cluster_listings)

        # If listings match user preferences, recommend the highest-rated listing
        if not cluster_listings.empty:
            recommended_listing = cluster_listings.loc[cluster_listings['rating'].idxmax()]
            recommended_id = recommended_listing['id']  # Accessing the ID of the recommended listing
            return recommended_listing, total_filtered_listings, recommended_id

    # If no suitable listings were found in any cluster
    print("No suitable listing found in any cluster.")
    return None, total_filtered_listings, None


def main(in_directory, max_budget, search_range, room_type_input, min_rating):
    data = pd.read_csv("data/listings.csv")
    
    # If the 'rating' column already exists, just fill missing values.
    # This replaces the previous method of re-extracting ratings.
    if 'rating' in data.columns:
        mean_rating = data['rating'].dropna().mean()
        data.loc[data['rating'].isnull(), 'rating'] = mean_rating  # Avoids chained assignment warning.
    else:
        # Extract ratings if the column doesn't exist.
        data['rating'] = data['name'].apply(extract_rating).fillna(data['rating'].dropna().mean())

    # Proceed with clustering without including room_type in the ML model
    data, kmeans_model = cluster_listings(data)

    # Get coordinates from the photo
    lat, lon = get_coordinates(photo=in_directory)
    
    user_preferences = {
        'max_budget': float(max_budget) if max_budget.lower() != "all" else float('inf'),
        'min_rating': float(min_rating),
        'search_range': float(search_range),
        # Room type preference is no longer used in clustering or recommendation logic here
    }
    
    recommended_listing, total_filtered_listings, recommended_id = recommend_listing(data, lat, lon, search_range, user_preferences)

    if recommended_listing is not None:
        print(f"Recommended Listing: {recommended_listing['name']}\n")
    else:
        print("No suitable listing found.")

    final_filtered_listings = data[(data['distance'] <= search_range) &
                                    (data['price'] <= float(max_budget) if max_budget.lower() != "all" else data['price']) &
                                    (data['rating'] >= min_rating)].copy()
    
    final_filtered_listings['distance_to_user'] = final_filtered_listings.apply(
        lambda row: haversine(lon, lat, row['longitude'], row['latitude']), axis=1
    )

    final_filtered_listings.sort_values(by='distance_to_user', inplace=True)
    final_filtered_listings.to_csv('results/hotel.csv', index=False)
    
    amenity_data = pd.read_json('data/amenities-vancouver.json.gz', lines=True)
    categorized_restaurants = categorize_restaurants(amenity_data)
    
    prepare_and_display_airbnb_listings(data, lat, lon, max_budget, search_range, room_type_input, min_rating, amenity_data, recommended_id)
    display_chain_non_chain_restaurants(categorized_restaurants, lat, lon, search_range, amenity_data)

if __name__ == '__main__':
    args = sys.argv[1:]
    in_directory = args[0] if len(args) > 0 else input("Enter the directory of the photo: ")
    max_budget = args[1] if len(args) > 1 else input("Enter maximum budget (or 'all' for no limit): ")
    search_range = int(args[2] if len(args) > 2 else input("Enter search range in meters (e.g., 300): "))
    room_type_input = args[3] if len(args) > 3 else input("Enter room type (1 for Entire home/apt, 2 for Private room, any other key for no preference): ")
    min_rating = float(args[4] if len(args) > 4 else input("Enter minimum rating (0-5, use 0 for no minimum): "))

    main(in_directory, max_budget, search_range, room_type_input, min_rating)
