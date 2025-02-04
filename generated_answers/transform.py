import re


def remove_non_code(input_text):
    # Regex to match code blocks and capture the content after any optional language hint
    code_block_pattern = r"```(\w+)?\s*(.*?)```"
    
    # Extract all code blocks and clean them
    matches = re.findall(code_block_pattern, input_text, re.DOTALL)
    
    # Get only the content inside the code block
    transformed_matches = [match[1].strip() for match in matches]
    
    # Join the results with newline
    return "\n".join(transformed_matches)


def strip_and_truncate(text, coding_task=False):
    """
    If coding_task=True, for each function definition in 'text' (including multiple), 
      - identifies the function name between 'def ' and '(' 
      - removes any triple-quoted docstring that immediately follows the function signature
    Otherwise, returns the original text unchanged.

    The docstring is defined as a block of \"\"\"...\"\"\" or '''...''' 
    that appears right after the function signature.

    Args:
        text (str): The entire code or text snippet.
        coding_task (bool): If True, perform removal of docstrings; if False, do nothing.

    Returns:
        str: The updated text, with function docstrings removed if coding_task=True.
    """
    if not coding_task:
        return text

    # Regex: 
    # (1) Capture the function signature (group 1) => "def foo(...) :" including whitespace 
    # (2) Sub-capture the function name (group 2) => used if you want to log or store
    # (3) Capture the triple-quoted docstring (group 3) => We'll remove this part
    pattern = r'(def\s+([A-Za-z_]\w*)\s*\(.*?\):\s*)(["\']{3}[\s\S]*?["\']{3})'

    # We use a while loop to repeatedly remove docstrings from each occurrence, 
    # including multiple functions or multiple docstrings in a row.
    text_out = text
    while True:
        match = re.search(pattern, text_out, flags=re.DOTALL)
        if not match:
            break
        # docstring = match.group(3)  # if you want to see or log it
        prefix = match.group(1)
        # We'll keep the function signature, remove the docstring:
        text_out = re.sub(pattern, prefix, text_out, count=1, flags=re.DOTALL)

    return text_out



input_1 =   "It looks like you're starting to write a function in Python that might simulate or calculate something related to car race collisions. To help you, I'll provide a simple example of how you might structure such a function. However, I'll need to make some assumptions about what you're trying to achieve since the function's purpose isn't fully clear. Let's assume you want to determine if a collision occurs during a race based on positions of cars.  Here\u2019s a basic example to get you started:  ```python def car_race_collision(positions):     \"\"\"     Determines if a collision occurs in a car race based on car positions.      :param positions: A list of integers representing the positions of cars on the track.                       Each index represents a car, and the value at each index represents                       the position of that car on the track.     :return: A boolean indicating whether a collision has occurred (True if a collision occurs, False otherwise).     \"\"\"     # Create a set to track unique positions     position_set = set()      # Iterate over the list of positions     for position in positions:         # Check if the position is already in the set         if position in position_set:             # Collision occurs if a position is repeated             return True         # Add the current position to the set         position_set.add(position)      # If no collisions are found, return False     return False  # Example usage: car_positions = [0, 1, 2, 3, 2]  # The last car is at the same position as another car print(car_race_collision(car_positions))  # Output: True ```  ### Explanation: - **Function Input**: The function takes a list of integers, where each integer represents a car's position on the track. - **Collision Detection**: The function uses a set to track positions that have already been seen. If a position is encountered more than once, a collision is detected. - **Return Value**: The function returns `True` if a collision is detected and `False` otherwise.  This is a simple example, and you might want to expand it based on your specific needs, such as considering more complex rules for collisions, handling multiple laps, or simulating movement over time. If you provide more specific requirements, I can help refine the function further!"
o1 =        'def car_race_collision(positions):     """     Determines if a collision occurs in a car race based on car positions.      :param positions: A list of integers representing the positions of cars on the track.                       Each index represents a car, and the value at each index represents                       the position of that car on the track.     :return: A boolean indicating whether a collision has occurred (True if a collision occurs, False otherwise).     """     # Create a set to track unique positions     position_set = set()      # Iterate over the list of positions     for position in positions:         # Check if the position is already in the set         if position in position_set:             # Collision occurs if a position is repeated             return True         # Add the current position to the set         position_set.add(position)      # If no collisions are found, return False     return False  # Example usage: car_positions = [0, 1, 2, 3, 2]  # The last car is at the same position as another car print(car_race_collision(car_positions))  # Output: True'
o2 =        'def car_race_collision(positions):     \"\"\"     Determines if a collision occurs in a car race based on car positions.      :param positions: A list of integers representing the positions of cars on the track.                       Each index represents a car, and the value at each index represents                       the position of that car on the track.     :return: A boolean indicating whether a collision has occurred (True if a collision occurs, False otherwise).     \"\"\"     # Create a set to track unique positions     position_set = set()      # Iterate over the list of positions     for position in positions:         # Check if the position is already in the set         if position in position_set:             # Collision occurs if a position is repeated             return True         # Add the current position to the set         position_set.add(position)      # If no collisions are found, return False     return False  # Example usage: car_positions = [0, 1, 2, 3, 2]  # The last car is at the same position as another car print(car_race_collision(car_positions))  # Output: True'
# Example usage
output_1 = remove_non_code(input_1)
output_1 = strip_and_truncate(output_1)
print(output_1)
