Thanks for checking out my project! 

Included are two folders, as the development took place in two places.

In ubuntu, I trained the models using tensorflow, quantized the weights, and stored the model information as a hex file for vivado. In this folder is also a short script for analyzing the output of the model when ran in simulation using the testbench.sv file.

The other folder is the Vivado Project File, which contains all the SystemVerilog files and Basys3 constraints. It should just open in Vivado with no struggle (well, no more struggle than is usual for Vivado). This also contains a testbench, which tests all the values in a given set of inputs, and saves the input/output pairs to a file for comparison to the purely software model from tensorflow. SystemVerilog I wrote can be found in MLP/MLP.srcs, and the testbench output file can be found in MLP\MLP.sim\sim_1\behav\xsim saved as Neuron_Test1_Output.txt. A copy of this file has been saved in the Ubuntu folder as well for an easy comparison.