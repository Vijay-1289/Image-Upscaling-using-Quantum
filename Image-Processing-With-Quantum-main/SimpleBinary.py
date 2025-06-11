matrix = [[0, 100, 24],
          [38, 43, 55],
          [62, 17, 255]]

def binaryConversion(n):
    return f'{n:08b}'

def average_to_binary(matrix):
    total = 0
    count = 0
    binary_matrix = []

    for row in matrix:
        binary_row = []
        for num in row:
            binary = binaryConversion(num)
            binary_row.append(binary)
            total += num
            count += 1
        binary_matrix.append(binary_row)

    average = total // count
    average_binary = binaryConversion(average)

    return binary_matrix, average, average_binary

# Get binary matrix and average
binary_matrix, avg_decimal, avg_binary = average_to_binary(matrix)

# Find min and max values in the original matrix
max_val = max(max(row) for row in matrix)
min_val = min(min(row) for row in matrix)

def binaryBinarization(avg_val, mat, min_val, max_val):
    result = []
    for row in mat:
        row_result = []
        for val in row:
            if val >= avg_val:
                row_result.append(binaryConversion(max_val))  # Bright
            else:
                row_result.append(binaryConversion(min_val))  # Dark
        result.append(row_result)
    return result

# --- Run ---
print("Original Matrix:")
for row in matrix:
    print(row)

print("\nBinary Matrix:")
for row in binary_matrix:
    print(row)

print(f"\nAverage (decimal): {avg_decimal}")
print(f"Average (binary): {avg_binary}")
print(f"Max value (decimal): {max_val}, binary: {binaryConversion(max_val)}")
print(f"Min value (decimal): {min_val}, binary: {binaryConversion(min_val)}")

print("\nBinarized Matrix:")
binarized_result = binaryBinarization(avg_decimal, matrix, min_val, max_val)
for row in binarized_result:
    print(row)
