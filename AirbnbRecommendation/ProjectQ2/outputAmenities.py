import gzip
import json
from collections import Counter

# Input file - Adjust the filename to match your output file
input_filename = 'output/part-00000-ebf78c7b-ec71-4a1a-950e-51bb732273e5-c000.json.gz'
# Output file
output_filename = 'output/outputTags.txt'

# Initialize a Counter object to count the occurrences of each amenity
amenities_counter = Counter()

with gzip.open(input_filename, 'rt', encoding='utf-8') as infile:
    for line in infile:
        data = json.loads(line)
        amenity = data.get('amenity', None)
        if amenity:
            amenities_counter[amenity] += 1

# Write the counts to the output file
with open(output_filename, 'w', encoding='utf-8') as outfile:
    for amenity, count in amenities_counter.items():
        outfile.write(f'{amenity}\t{count}\n')

print(f'Amenity counts have been written to {output_filename}')