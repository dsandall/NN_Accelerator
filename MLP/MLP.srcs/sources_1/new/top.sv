`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/27/2024 11:58:02 AM
// Design Name: 
// Module Name: topWrapper
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
`include "defs.svh"

module topWrapper(
    input CLK, 
    input signed [BITS_INPUT-1:0] input_x,
    output signed [BITS_OUTPUT-1:0] output_y
);

    logic signed [BITS_OUTPUT-1:0] hidden_input [0:0];    
    assign hidden_input [0] = input_x;
    reg signed [BITS_BIAS-1:0] hidden_biases [NUM_HIDDEN-1:0];
    reg signed [BITS_WEIGHT-1:0] hidden_weights [NUM_HIDDEN-1:0] [0:0]; //number of neurons, number of weights per neuron 
   
    logic signed [BITS_OUTPUT-1:0] hidden_output [NUM_HIDDEN-1:0];

    reg signed [BITS_BIAS-1:0] output_biases[0:0];
    reg signed [BITS_WEIGHT-1:0] output_weights [0:0] [NUM_HIDDEN-1:0]; //number of neurons, number of weights per neuron
    reg signed [BITS_WEIGHT-1:0] output_output [0:0];


    wire [BITS_OUTPUT-1:0] out;


    // parse through the weights and biases
    initial begin
        integer file, r;
        string line;
        int addr;
        int data;
        int increment;
        file = $fopen(HIDDEN_LAYER_MEM, "r");
        if (file == 0) begin
            $display("Error: Could not open file %s", HIDDEN_LAYER_MEM);
            $finish;
        end

        addr = 0;
        while (!$feof(file)) begin
            r = $fgets(line, file);
            if (line.len() <= 0) begin
            end
            else if (line[0] == "#") begin
                addr = 0;
            end
            else begin
                r = $sscanf(line, "%x", data);

                if (r == 1) begin
                    if (increment <= NUM_HIDDEN-1) begin
                        hidden_weights[addr][0] = data;
                        end
                    if (increment > NUM_HIDDEN-1 && increment <= ((NUM_HIDDEN * 2)-1)) begin
                        hidden_biases[addr] = data;
                        end
                    if (increment > ((NUM_HIDDEN * 2)-1) && increment <= ((NUM_HIDDEN * 3)-1)) begin 
                        output_weights[0][addr] = data;
                        end
                    if (increment > ((NUM_HIDDEN * 3)-1) && increment <= ((NUM_HIDDEN * 4)-1)) begin
                        output_biases[0] = data;
                        end
                    increment++;
                    addr++;
                end
            end
        end

        $fclose(file);
    end

    

    // Instantiate Hidden Neurons
    generate
        genvar i;
        for (i = 0; i < NUM_HIDDEN; i = i + 1) begin : hidden_neurons
            neuron2 #(
                .ACT_FUNC_MEM("sigmoid_function.mem"), 
                .INPUT_CNT(1)
            ) hidden (
                .bias(hidden_biases[i]),
                .w(hidden_weights[i]),
                .x(hidden_input),
                .clk(CLK),
                .y(hidden_output[i])
            );
        end
    endgenerate

    // Instantiate Output Neuron
        // Instantiate Hidden Neurons
    generate
        genvar ii;
        for (ii = 0; ii < NUM_OUTPUT; ii = ii + 1) begin : output_neurons
            neuron2 #(
                .ACT_FUNC_MEM(""), 
                .INPUT_CNT(NUM_HIDDEN)
            ) output_neuron (
                .bias(output_biases[ii]),
                .w(output_weights[ii]),
                .x(hidden_output),
                .clk(CLK),
                .y(output_output[ii])
            );
        end
    endgenerate


    assign output_y = output_output[0];

endmodule




