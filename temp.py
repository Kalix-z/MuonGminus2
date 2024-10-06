import random

# Function to remove random lines from the input file
def remove_random_lines(input_file, num_lines_to_remove):
    # Read all lines from the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Check if there are enough lines to remove, excluding the first line
    if num_lines_to_remove >= len(lines) - 1:  # Exclude the first line
        print("Error: Number of lines to remove exceeds the total number of lines (excluding the first line).")
        return

    # Randomly select lines to remove (excluding the first line)
    lines_to_remove = random.sample(range(1, len(lines)), num_lines_to_remove)  # Start from index 1

    # Remove the selected lines
    remaining_lines = [line for i, line in enumerate(lines) if i == 0 or i not in lines_to_remove]

    # Write the remaining lines back to a new file or overwrite the original
    with open('COSYinput_modified.txt', 'w') as output_file:
        output_file.writelines(remaining_lines)

    print(f"Removed {num_lines_to_remove} lines. {len(remaining_lines)} lines remain.")

# Specify the input file and number of lines to remove
input_file = 'COSYinput.txt'
num_lines_to_remove = 5000

# Call the function
remove_random_lines(input_file, num_lines_to_remove)
