'''2. Generate ANDNOT function using McCulloch-Pitts neural net by a python program.
'''

def andnot_function(x1, x2, w1, w2, theta=1):
    """
    Perform the ANDNOT operation using the McCulloch-Pitts neural network model.

    Parameters:
    x1, x2: Lists of inputs.
    w1, w2: Weights for inputs.
    theta: Threshold value.

    Returns:
    List of outputs and a message indicating if the function is correct.
    """
    # Calculate the output list
    output = []
    for i in range(4):
        # Calculate weighted sum
        weighted_sum = x1[i] * w1 + x2[i] * w2

        # Determine output based on threshold
        if weighted_sum >= theta:
            output.append(1)
        else:
            output.append(0)

    # Define the expected output
    expected_output = [0, 0, 1, 0]

    # Check if the output matches the expected output
    is_correct = output == expected_output

    # Print the results and correctness message
    print("Output:", output)
    if is_correct:
        print("ANDNOT function is correct for all inputs!")
    else:
        print("ANDNOT function is not correct!")

# Test the function with provided weights and threshold
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
w1 = float(input("Enter w1: "))
w2 = float(input("Enter w2: "))

andnot_function(x1, x2, w1, w2)
