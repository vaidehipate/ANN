'''Write a Python Program using Perceptron Neural Network to recognise even and odd numbers.
Given numbers are in ASCII form 0 to 9
'''


import numpy as np

# Define the input-output mapping
inputs_array = np.array([[0,0,1,1,0,0,0,0],
                      [0,0,1,1,0,0,0,1],
                      [0,0,1,1,0,0,1,0],
                      [0,0,1,1,0,0,1,1],
                      [0,0,1,1,0,1,0,0],
                      [0,0,1,1,0,1,0,1],
                      [0,0,1,1,0,1,1,0],
                      [0,0,1,1,0,1,1,1],
                      [0,0,1,1,1,0,0,0],
                      [0,0,1,1,1,0,0,1]])
expected_output = np.array([[1],[0],[1],[0],[1],[0],[1],[0],[1],[0]])

# Get user input
while True:
    user_input = input("Enter one of the following numbers: 0, 1, 2, 3, 4, 5, 6, 7, 8\n")
    if user_input.isdigit() and int(user_input) in range(9):
        user_input = int(user_input)
        break
    else:
        print("Invalid input. Please enter a number between 0 and 8.")

# Define weights and bias
weights = 1
bias = 2
sum = 0

epoch = 15
for j in range(epoch):
    predicted = 0
    for i in range(len(inputs_array[user_input])):
        sum += inputs_array[user_input][i] * weights
    output = sum + bias

    # Step function
    if output > 0:
        predicted = 1
        print("The number is even.")
    else:
        predicted = 0
        print("The number is odd.")

    print("Expected ", expected_output[user_input])
    if expected_output[user_input] == predicted:
        break
    else:
        for k in range(len(inputs_array[user_input])):
            weights = weights + ((0.1) * (expected_output[user_input] - predicted) * inputs_array[user_input][k])
            print("Weights update ", weights)
        print("Weights", weights)
