import matplotlib.pyplot as plt
import numpy as np




def main():

    # Read the filename from the user
    filename = 'Neuron_Test1_Output.txt'

    # Parse the file
    inputs, outputs, input_flags, output_flags = parse_file(filename)

    # Plot the data
    plot_data(inputs, outputs, input_flags, output_flags)


# define the function that is modelled by the NN
x_min = 0
x_max = 5    
def func(x):
    return x * np.exp(-x)





##### Helper Functions #####

def parse_q4_6(hex_value):
    """Convert a Q4.6 hex value to a decimal. Return a tuple with the value and a flag indicating if it's an 'XXX'."""
    if hex_value == 'XXX' or hex_value == 'xxx':
        return 0, True
    else:
        # Convert hex string to an integer
        int_value = int(hex_value, 16)
        
        # Handle negative values
        if int_value & 0x200:  # Check the sign bit (bit 9)
            int_value -= 0x400  # Apply two's complement for negative numbers
        
        # Convert to Q4.6
        return int_value / 64.0, False

def parse_file(filename):
    """Parse the file and return input and output lists along with flags for 'XXX' values."""
    inputs = []
    outputs = []
    input_flags = []
    output_flags = []
    
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            input_val = parts[1].split(': ')[1]
            output_val = parts[2].split(': ')[1]
            
            input_dec, input_flag = parse_q4_6(input_val)
            output_dec, output_flag = parse_q4_6(output_val)
            
            inputs.append(input_dec)
            outputs.append(output_dec)
            input_flags.append(input_flag)
            output_flags.append(output_flag)
    
    return inputs, outputs, input_flags, output_flags






def plot_data(inputs, outputs, input_flags, output_flags):
    """Plot the input and output values, highlighting 'XXX' values in red."""
    plt.figure(figsize=(10, 6))

    # plot desired function
    x_test = np.linspace(x_min, x_max, 100)
    y_test = func(x_test)
    plt.plot(x_test, y_test, 'g', label='Desired Function')

    
    # Plot input vs output values
    plt.plot(inputs, outputs, label='Input vs Output (Q4.6)', color='blue', marker='o', linestyle='none')
    
    # Highlight 'XXX' values in red
    plt.plot([inputs[i] for i in range(len(inputs)) if input_flags[i] or output_flags[i]], 
             [outputs[i] for i in range(len(outputs)) if input_flags[i] or output_flags[i]], 
             'ro')
    
    plt.xlabel('Input (Q4.6)')
    plt.ylabel('Output (Q4.6)')
    plt.title('NN Input vs Output for xe^(-x)')
    plt.legend()
    plt.grid(True)
    
    plt.show()





if __name__ == '__main__':
    main()