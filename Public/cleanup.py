import re

# Define the input and output file paths
input_file_path = 'PDFMalware2022.csv'
output_file_path = 'PDFMalware2022Grepped.csv'

# Define the file containing the patterns to exclude
exclude_file_path = 'ranges2'

# Open the input and exclude files
with open(input_file_path, 'r') as input_file, open(exclude_file_path, 'r') as exclude_file:
    # Read the exclude file into a list of patterns
    exclude_patterns = exclude_file.readlines()
    # Remove newline characters from the patterns
    exclude_patterns = [re.escape(pattern.rstrip('\n')) for pattern in exclude_patterns]

    # Open the output file
    with open(output_file_path, 'w') as output_file:
        # Loop through the lines in the input file
        for line in input_file:
            # Check if the line matches any of the patterns to exclude
            if not any(re.search(pattern, line) for pattern in exclude_patterns):
                # Write the line to the output file
                output_file.write(line)


import re

# Define the input file path
input_file_path = 'PDFMalware2022Grepped.csv'

# Define the regular expression pattern to match
pattern = r'\([0-9]\)'

# Define the replacement string
replacement = ''

# Open the input file
with open(input_file_path, 'r') as input_file:
    # Read the entire contents of the file into a string
    input_string = input_file.read()

    # Use the re.sub() function to perform the substitution
    output_string = re.sub(pattern, replacement, input_string)

# Open the input file again, this time for writing
with open(input_file_path, 'w') as output_file:
    # Write the modified string to the file
    output_file.write(output_string)

