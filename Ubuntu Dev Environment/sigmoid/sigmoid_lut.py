import math 
# from fixedpoint import FixedPoint

# # Define Q4.6 format (total 10 bits, 6 fractional bits)
# class FixedPointInput(FixedPoint):
#     def __new__(cls, value):
#         return FixedPoint.__new__(cls, value, signed=True, m=4, n=6, str_base=10)
    
# # Define Q4.6 format (total 10 bits, 6 fractional bits)
# class FixedPointOutput(FixedPoint):
#     def __new__(cls, value):
#         return FixedPoint.__new__(cls, value, signed=True, m=4, n=6, str_base=10)

fractional_intput_bits = 6
fractional_output_bits = 6

def sigmoid(x):
  """Calculates the sigmoid function value for a given input."""
  return 1 / (1 + math.exp(-x))

def ints_to_hex(sigmoid_ints):
  return [f"{val:05x}" for val in sigmoid_ints]


def generate_sigmoid_fixed_as_int(inputres, outputres):
  sigmoid_ints = [int(round(2**fractional_output_bits * sigmoid(fixedtofloat(x)))) for x in range(-(2**(inputres-1)), (2**(inputres-1))-1)]
  print("{:05X}".format(sigmoid_ints[0]))
  print("{:05X}".format(sigmoid_ints[-1]))
  print(fixedtofloat(sigmoid_ints[0]))
  print(fixedtofloat(sigmoid_ints[-1]))
  return sigmoid_ints


def write_to_file(filename, data):
  """Writes the provided data to a text file."""
  with open(filename, 'w') as f:
    f.write('\n'.join(data))



# def normalize_signed_int(value, resolution):
#   """Normalizes an 8-bit signed integer to the range -1 to 1."""
#   scaling_factor = 2
#   scaled_value = value * scaling_factor
#   normalized_value = scaled_value / float((2**resolution)-1)  # Use float for division
#   return normalized_value



def percent_diff(actual, expected):
  return abs((actual - expected)/expected)



fractionalbits = 6
def fixedtofloat(fixed):
  return fixed / (2**fractionalbits)

def floattofixed(fixed):
  return fixed * 2**fractionalbits
  

# Set the desired input resolution (adjust as needed)
# these are assumed to be signed !
# NOTE: you may need to adjust the number of hex chars 
outputres = 10 # the resolution of the table value in bits
inputres = 10 # the resolution of the table input (index) in bits



# for 10 bit values, minimum integer value should be -2^9, and max should be (2^9)-1 (-512 to 511)
inp_range = range(-(2**(inputres-1)), (2**(inputres-1))-1)
out_range = range(-(2**(outputres-1)), (2**(outputres-1))-1)

sigmoid_fixed_ints = generate_sigmoid_fixed_as_int(inputres, outputres)

print(inp_range)
print(out_range)
print("min input", fixedtofloat(inp_range[0]))         
print("max input", fixedtofloat(inp_range[-1]))        
print("min sigmoid output", fixedtofloat(sigmoid_fixed_ints[0]))     #this should come out to 0
print("max sigmoid output", fixedtofloat(sigmoid_fixed_ints[-1]))    #this should come out to slightly less than 1

avg_percent_diff = sum(percent_diff(fixedtofloat(sigmoid_fixed_ints[i]), sigmoid(fixedtofloat(inp_range[i]))) for i in range(len(inp_range))) / len(inp_range)
max_percent_diff = max(percent_diff(fixedtofloat(sigmoid_fixed_ints[i]), sigmoid(fixedtofloat(inp_range[i]))) for i in range(len(inp_range)))

print("avg percent difference", avg_percent_diff)
print("max", max_percent_diff)
print("percent differences",
      percent_diff(fixedtofloat(inp_range[0]), fixedtofloat(inp_range[0])),
      percent_diff(fixedtofloat(inp_range[-1]), fixedtofloat(inp_range[-1])),
      percent_diff(fixedtofloat(sigmoid_fixed_ints[0]), sigmoid(fixedtofloat(inp_range[0]))),
      69, 
      fixedtofloat(sigmoid_fixed_ints[100]), 
      sigmoid(fixedtofloat(inp_range[100])),
      79,
      percent_diff(fixedtofloat(sigmoid_fixed_ints[-100]), sigmoid(fixedtofloat(inp_range[-100]))),
      percent_diff(fixedtofloat(sigmoid_fixed_ints[-1]), sigmoid(fixedtofloat(inp_range[-1])))
      )

# Generate the sigmoid function values in hex

# def generateLookup(value, inputFormat, outputFormat):
#   (bits, fractionalBits) = inputFormat
#   (outBits, outFractionalBits) = outputFormat


# generateLookup(.78125, (10, 6), (10, 6))

# Specify the output filename
filename = "sigmoid_lookup.txt"
sigmoid_hex_list = ints_to_hex(sigmoid_fixed_ints)
# Write the hex values to the file
write_to_file(filename, sigmoid_hex_list)

print(f"Sigmoid function lookup table saved to: {filename}")
