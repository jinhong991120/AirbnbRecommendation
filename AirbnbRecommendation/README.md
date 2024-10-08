# Airbnb Recommendation Program

This project is an Airbnb recommendation system designed to help users find the best accommodation options in the Greater Vancouver area based on their preferences. The system utilizes various data points and implements machine learning clustering algorithms to provide personalized suggestions.

## Features

- **Personalized Recommendations**: Users receive accommodation suggestions based on specific input criteria such as budget, preferred room type, minimum rating, and search radius from a given location.
- **Location-Based Filtering**: Utilizing the user's current location, the system filters out listings that fall within the specified search range.
- **Rating Extraction and Processing**: The system extracts ratings from listing names, fills missing values, and ensures data consistency for better recommendation accuracy.
- **Clustering of Listings**: Implements KMeans clustering to categorize listings for more nuanced recommendations.
- **Visualization**: Integrates Folium maps to display Airbnb listings and nearby amenities, enhancing user experience through visual exploration of options.
- **Distance Calculation**: Employs the Haversine formula to calculate distances between points of interest, ensuring relevance in recommendations.

## Instructions

Before running the recommendation system,navigate to the project directory ~/CMPT353/ProjectQ2 after cloning. Ensure you have the following prerequisites installed:
- Python 3.x
- Libraries: 
```shell
Python: Primary programming language used.
Pandas: Data manipulation and analysis.
Folium: Interactive map visualizations.
Sklearn: Machine learning algorithms of clustering.
ExifRead: Metadata extraction from images.
NumPy: Numerical operations.
Re: Regular expressions of data extraction.
```
 - Install the required libraries by typing `pip install -r requirements.txt` in the terminal.

## Usage

The system is executed via a command-line interface. To start the program, navigate to the project directory (~/CMPT353/ProjectQ2) and run:

```shell
python3 finder.py [photo directory] [max budget] [search range] [room type] [min rating]
```
Program takes 5 parameters following:
- [photo directory]: Directory of the photo to extract the user's current location.
- [max budget]: Maximum budget for accommodation, or 'all' for no limit.
- [search range]: Search radius in meters.
- [room type]: Preferred room type ('1' for Entire home/apt, '2' for Private room).
- [min rating]: Minimum acceptable rating (0-5).

## Demo Example
I have included 3 demo photos in photo folder. User can replace input photo that is taken with GPS turned on within Greater Vancouver. Make sure you are on the correct directory. (~/CMPT353/ProjectQ2) and type
```shell
python3 finder.py photo/1.JPG 200 200 1 4
```
Above command line will save and generate information of listings of Airbnb and restaurants in html file, and automatically pops 2 new web-browser map for visualization. Detailed information about the map is described in report part VI, page 4.

## Folder Structure

Below is the folder structure/description for the project:

```plaintext
.
PROJECTQ2
├── data
│   ├── amenities-vancouver.json.gz        # Given osm data file in json format
│   ├── list_of_chain_restaurants.csv      # List of chains in Greater Vancouver in csv format
│   ├── listings.csv                       # List of airbnb listings in csv format
│   └── reviews.csv                        # List of airbnb reviews in csv format
│
├── osm
│   └── code                               # Provided python codes by sbergner
│       ├── disassemble-osm.py
│       └── osm-amenities.py
│       └── just-vancouver.py
│
├── output
│   ├── part-00000-ebf78c7b-...json        # Output of osm data by just-vancouver.py 
│   ├── outputAll.txt                      # Output of every row and column from above json file in text
│   ├── outputAmenities.txt                # Output of amenities along with their repetitive number
│   ├── outputAmenTags.txt                 # Output of amenities and tags for each row
│   └── outputTags.txt                     # Output of tags along with their repetitive number for each row
│
├── photo
│   ├── 1.JPG                              # Demo pictures
│   ├── 2.JPG
│   └── 3.JPG
│
├── results
│   ├── airbnb_listings_map.html           # Visualization map with airbnb icons and heatmap
│   ├── hotel.csv                          # Airbnb on map information in csv file
│   ├── map.html                           # Visualization of map
│   ├── restaurants_map.html               # Visualization map with restaurant icons
│   ├── Figure1.jpg                        # Distribution of Restaurants in Greater Vancouver
│   ├── Figure2.1.jpg                      # Density of Chain Restaurants in Greater Vancouver
│   ├── Figure2.2.jpg                      # Density of Non-Chain Restaurants in Greater Vancouver
│   ├── Figure3.jpg                        # Histogram of Amenities Frequency in Vancouver
│   ├── Figure4.jpg                        # Density of Good Amenities in Greater Vancouver
│   ├── Figure5.jpg                        # Density of Good Amenities vs Distance from Downtown Vancouver
│
├── finder.py                              # Main program
├── figures.ipynb                          # Figure generating program in jupyter notebook 
├── outputAll.py                           # Python code for outputAll.txt         
├── outputAmenities.py                     # Python code for outputAmenities.txt     
├── outputAmenTags.py                      # Python code for outputAmenTags.txt     
├── outputTags.py                          # Python code for outputTags.txt     
└── requirements.txt                       # Text file for pip install external libraries

