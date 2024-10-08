import gzip
import json

# Input file - Adjust the filename to match your output file
input_filename = 'output/part-00000-ebf78c7b-ec71-4a1a-950e-51bb732273e5-c000.json.gz'
# Output file
output_filename = 'output/outputAmenTags.txt'

with gzip.open(input_filename, 'rt', encoding='utf-8') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        # Extract amenities and tags
        amenity = data.get('amenity', 'N/A')  # Default to 'N/A' if amenity is missing
        tags = data.get('tags', {})
        # Format output string to include only amenities and tags
        output_line = f'{amenity}\t{json.dumps(tags)}\n'  # Use tab to separate amenity and tags, and convert tags dict back to JSON string
        # Write to output file
        outfile.write(output_line)

print(f'Data has been written to {output_filename}')