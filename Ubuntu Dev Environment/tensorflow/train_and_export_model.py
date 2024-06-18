import tensorflow as tf
import numpy as np
from datetime import datetime
import os
from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot
from fxpmath import Fxp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# train new model, or use quantize and export existing model (model architecture must match)
trainNewModel = 0
model_to_load = "xe-x_PrettyGood.keras"
 
# set the number of training epochs
epochs = 100

# Define the function xe^-x and it's domain
x_min = 0
x_max = 5

def func(x):
    return x * tf.exp(-x)

# Q4.6 format numbers
num_bits = 10
frac_bits = 6

def main():
    # Generate some training data
    x_train = np.linspace(x_min, x_max, 5000)
    y_train = func(x_train) 

    # Define the regular model
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(1,)),
        keras.layers.Dense(5, activation='sigmoid'),  #sigmoid is much better for linear functions and for small networks
        # keras.layers.Dense(5, activation='relu'), 
        # keras.layers.Dense(5, activation='relu'),
        keras.layers.Dense(1)
    ])

    
    if (trainNewModel):

        model = TrainRegular(model, x_train, y_train)
        # q_model = TrainQuantized(model, x_train, y_train)

        # CHECKPOINT MODEL
        save_model_with_timestamp(model)
        
    else:

        project_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(project_dir, 'saved_models')
        
        # Get the desired model, or get the most recent model if not specified 
        loaded_model = load_model(save_dir)

        model = loaded_model

        # loaded_model.summary()

        # test_performance(loaded_model)


    # Evaluate the models
    x_test = np.linspace(x_min, x_max, 100)
    y_test = func(x_test)
    y_pred = model.predict(x_test)

    q_model = quantize_model_weights_fixed_point(model)
    q_y_pred = q_model.predict(x_test)


    # PLOT THE RESULTS
    import matplotlib.pyplot as plt
    plt.plot(x_test, y_test, 'b.', label='Training Data')
    plt.plot(x_test, q_y_pred, 'g-', label='quantized Predictions')
    plt.plot(x_test, y_pred, 'r-', label='Predictions')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    model.summary()

    # Print weights of each layer in the quantized model
    print("\nQuantized Model Weights:")

    i = 0
    for layer in q_model.layers:
        hex_weights = float_to_fixed_hex(layer.get_weights()[0].flatten())

        print("weights at layer ", i,  layer.get_weights()[0].flatten(), hex_weights)

        hex_weights = float_to_fixed_hex(layer.get_weights()[1].flatten())

        print("biases at layer ", i,  layer.get_weights()[1].flatten(), hex_weights)

        i = i + 1


    #write the weights to a parsable format
    write_weights_to_file(q_model, "model_info_newfunction.txt")



    ############################
    
    # Note: a quantization aware model is not actually quantized. Creating a quantized model is a separate step.
    
    # Note: saving/checkpointing models https://www.tensorflow.org/model_optimization/guide/quantization/training_comprehensive_guide#checkpoint_and_deserialize
    
    # Note: Custom Quantization Settings will require "experimental" settings https://www.tensorflow.org/model_optimization/guide/quantization/training_comprehensive_guide#experiment_with_quantization

    #############################




def test_performance(model):
    # Evaluate the model on the test data
    x_test = np.linspace(x_min, x_max, 100)
    y_test = func(x_test)
    test_loss = model.evaluate(x_test, y_test)
    print(f'Test Loss: {test_loss}')
    # print(f'Test MAE: {test_mae}')

    # Predict on the test data
    y_pred = model.predict(x_test)

    # Calculate additional performance metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    # print(f'R^2 Score: {r2}')




callback = keras.callbacks.EarlyStopping(monitor='loss', patience=40)
def TrainRegular(model, x_train, y_train):
    # # Compile the model
    # optimizer = keras.optimizers.Adam(learning_rate=0.001)
    # optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
    # optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    # optimizer = keras.optimizers.Adagrad(learning_rate=0.01)
    # optimizer = keras.optimizers.Adadelta(learning_rate=1.0)
    optimizer = keras.optimizers.Nadam(learning_rate=0.005)

    model.compile(optimizer=optimizer, loss='mse')

    # Fit the model
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        callbacks=[callback],
        verbose=1
    )

    return model


def quantize_weights_fixed_point(weights, num_bits=num_bits, frac_bits=frac_bits):
    # Create a fixed-point object with the specified number of integer and fractional bits
    fxp = Fxp(None, signed=True, n_word=num_bits, n_frac=frac_bits)
    
    # Quantize weights
    quantized_weights = fxp(weights)
    
    # Dequantize weights back to floating point
    dequantized_weights = quantized_weights.astype(float)
    
    return dequantized_weights

def quantize_model_weights_fixed_point(model):
    # Clone the model to ensure original is not modified
    cloned_model = tf.keras.models.clone_model(model)
    cloned_model.set_weights(model.get_weights()) 

    
    for layer in cloned_model.layers:
        if hasattr(layer, 'kernel'):
            # Quantize kernel weights
            kernel_weights = layer.get_weights()[0]
            quantized_kernel_weights = quantize_weights_fixed_point(kernel_weights)
            
            # Quantize bias weights if they exist
            if len(layer.get_weights()) > 1:
                bias_weights = layer.get_weights()[1]
                quantized_bias_weights = quantize_weights_fixed_point(bias_weights)
                # Set the quantized weights back to the layer
                layer.set_weights([quantized_kernel_weights, quantized_bias_weights])
            else:
                # Set the quantized kernel weights back to the layer
                layer.set_weights([quantized_kernel_weights])
    
    return cloned_model



def save_model_with_timestamp(model, base_name='keras_model'):
    # Define the directory and filename for saving the model
    project_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(project_dir, 'saved_models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Generate a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create the model path with the timestamp
    model_path = os.path.join(save_dir, f'{base_name}_{timestamp}.keras')
    model.save(model_path)
    print(f'Model saved to: {model_path}')

def load_model(save_dir):
    global model_to_load

    if model_to_load is not None:
        model_to_load = model_to_load
    else:
        # List all files in the save directory
        files = [f for f in os.listdir(save_dir) if f.endswith('.keras')]
        
        # If no models are found, return None
        if not files:
            return None
        
        # Sort files by creation time
        files.sort(key=lambda x: os.path.getmtime(os.path.join(save_dir, x)), reverse=True)

        model_to_load = files[0]
    
    # Load the model
    most_recent_model_path = os.path.join(save_dir, model_to_load)
    with tfmot.quantization.keras.quantize_scope():
        loaded_model = keras.models.load_model(most_recent_model_path)
    
    print(f'Loaded model from: {most_recent_model_path}')
    return loaded_model


def float_to_fixed_hex(float_array, total_bits=10, fractional_bits=6):
    # Calculate scaling factor
    scaling_factor = 2 ** fractional_bits
    # Calculate max and min values for the given fixed-point format
    max_val = (2 ** (total_bits - fractional_bits - 1)) - (1 / scaling_factor)
    min_val = - (2 ** (total_bits - fractional_bits - 1))
    
    # Apply scaling factor and round to nearest integer
    fixed_array = np.round(float_array * scaling_factor)
    # Clip values to the range that can be represented
    fixed_array = np.clip(fixed_array, min_val * scaling_factor, max_val * scaling_factor)
    # Convert to integers
    fixed_int_array = fixed_array.astype(int)
    
    # Convert to hexadecimal representation
    hex_array = np.vectorize(lambda x: format(x & (2**total_bits - 1), '03x'))(fixed_int_array)
    return hex_array

def write_weights_to_file(q_model, filename):
    with open(filename, 'w') as file:
        file.write(f"# Model info:\n")
        file.write(f"# {model_to_load}\n")
        file.write(f"# generated at {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
        file.write(f"#\n")

        for i, layer in enumerate(q_model.layers):
            weights = layer.get_weights()[0].flatten()
            biases = layer.get_weights()[1].flatten()

            hex_weights = float_to_fixed_hex(weights)
            hex_biases = float_to_fixed_hex(biases)

            # Getting the number of neurons and the activation function
            num_neurons = layer.output_shape[-1]
            activation_function = layer.activation.__name__ if hasattr(layer, 'activation') else 'None'

            # Writing neurons and activation function
            file.write(f"#layer{i} neurons\n")
            file.write(f"#{num_neurons}\n")

            file.write(f"#layer{i} activation function\n")
            file.write(f"#{activation_function}\n")

            # Writing weights
            file.write(f"#layer{i} weights\n")
            for hex_weight in hex_weights:
                file.write(f"{hex_weight}\n")

            # Writing biases
            file.write(f"\n#layer{i} biases\n")
            for hex_bias in hex_biases:
                file.write(f"{hex_bias}\n")
            file.write("\n")






if __name__ == '__main__':
    main()