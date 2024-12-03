import re

input_file = "requirements.txt"
output_file = "cleaned_requirements.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Remove lines with build paths
        if "@ file://" in line:
            line = re.sub(r"\s*@\s*file://.*", "", line)
        # Write the cleaned line if it's valid
        if line.strip():
            outfile.write(line)

print(f"Cleaned requirements saved to {output_file}")

