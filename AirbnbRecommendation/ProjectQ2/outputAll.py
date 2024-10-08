import gzip
import json

# Input file - Adjust the filename to match your output file
input_filename = 'output/part-00000-ebf78c7b-ec71-4a1a-950e-51bb732273e5-c000.json.gz'
# Output file
output_filename = 'output/outputAll.txt'

with gzip.open(input_filename, 'rt', encoding='utf-8') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        # Convert the JSON object to a string format you desire
        output_line = json.dumps(data) + '\n'  # Example: convert back to JSON string
        # Write to output file
        outfile.write(output_line)

print(f'Data has been written to {output_filename}')
