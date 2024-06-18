`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/02/2024 12:06:39 AM
// Design Name: 
// Module Name: neuron
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

// // Function to multiply two fixed-point numbers
// function signed [TOTAL_BITS*2-1:0] fixed_mul;
//     input signed [TOTAL_BITS-1:0] a;
//     input signed [TOTAL_BITS-1:0] b;
//     reg signed [TOTAL_BITS*2-1:0] ab = a * b;
//     // fixed_mul =  >> FRAC_BITS;
//     fixed_mul = (ab[TOTAL_BITS*2-1]) || (ab[TOTAL_BITS*2-2 : 0]>>FRAC_BITS) ;
// endfunction

// // Function to divide two fixed-point numbers
// function signed [TOTAL_BITS-1:0] fixed_div;
//     input signed [TOTAL_BITS-1:0] a;
//     input signed [TOTAL_BITS-1:0] b;
//     fixed_div = (a << FRAC_BITS) / b;
// endfunction


// module neuron #(
//     parameter string ACT_FUNC_MEM = "", // Default: No memory file
//     parameter int INPUT_CNT = 5         // Number of inputs
// )(
//     input signed [BITS_BIAS-1:0] bias,
//     input signed [BITS_WEIGHT-1:0] w ,
//     input signed [BITS_INPUT-1:0] x ,
//     input clk ,
//     output reg [BITS_OUTPUT-1:0] y
// );
//     reg [BITS_OUTPUT-1 : 0] activation_LUT [0:1<<TOTAL_BITS]; // Activation LUT with 256 entries storing 8-bit data

//     // Load the LUT from the memory file if specified
//     initial begin
//         if (ACT_FUNC_MEM != "") begin
//             $display("Loading LUT from %s", ACT_FUNC_MEM);
//             $readmemh(ACT_FUNC_MEM, activation_LUT);
//         end
//     end

//     logic signed [TOTAL_BITS*2-1:0] wx;
//     logic signed [TOTAL_BITS*2-1:0] sum;
//     // logic signed [BITS_OUTPUT-1:0] out;
//     logic signed [TOTAL_BITS-1:0] roundedSum;
//     // logic truncated;


//     always @(posedge clk) begin
//         // Initialize the sum with bias

//         wx = w * x;
//         // sum = wx + bias;
//         // roundedSum = {wx[TOTAL_BITS-1], wx[FRAC_BITS+FRAC_BITS+INT_BITS-2:FRAC_BITS]}; //cut off the lower 6 and the upper 4, saving the sign bit
//         roundedSum = {wx[19], wx[14:6]};

//         roundedSum += bias;


//         //check for standard fixed point overflow
//         if ( wx < 0 && bias < 0 && roundedSum > 0)begin
//             roundedSum = -(1<<(TOTAL_BITS-1));
//         end else if ( wx > 0 && bias > 0 && roundedSum < 0) begin
//             roundedSum = ((1<<(TOTAL_BITS-1)))-1;
//         end

//         //check for out of bounds error (if sum is more than (1<<(TOTAL_BITS-1)), set roundedsum to max)
//         if (wx[18] != wx[19] || wx[17] != wx[19] || wx[16] != wx[19] || wx[15] != wx[19]) begin
//             if (wx[19] == 1) begin
//                 roundedSum = -(1<<(TOTAL_BITS-1));
//             end else begin
//                 roundedSum = ((1<<(TOTAL_BITS-1)))-1;
//             end
//         end


//         // roundedSum would be 1000 for -1, 000 for 0 , 01111 for 1
//         // the corresponding indeces would be 0 for -1, halfway for 0, max index for 1

//         // // if the sum is larger than the max for the standard fixed point value, set it to the max
//         // if (sum > (2 << TOTAL_BITS)) begin
//         //     out <= 10'b0111111111;
//         // end 
//         // else if (sum < () begin
//         // end 

        

//         if (ACT_FUNC_MEM != "") begin
//             y <= activation_LUT[roundedSum + ((1<<(TOTAL_BITS-1)))];
//         end else begin
//             y <= roundedSum; // Use sum directly if not using LUT
//         end
//     end

// endmodule

module neuron2 #(
    parameter string ACT_FUNC_MEM = "", // Default: No memory file
    parameter int INPUT_CNT = 5         // Number of inputs
)(
    input signed [BITS_BIAS-1:0] bias,
    input signed [BITS_WEIGHT-1:0] w [0:(INPUT_CNT-1)], // Handle INPUT_CNT = 1
    input signed [BITS_INPUT-1:0] x [0:(INPUT_CNT-1)],  // Handle INPUT_CNT = 1
    input clk,
    output reg [BITS_OUTPUT-1:0] y
);
    localparam int MIN_VAL = -(1<<(TOTAL_BITS-1));
    localparam int MAX_VAL = (1<<(TOTAL_BITS-1))-1;

    reg [BITS_OUTPUT-1 : 0] activation_LUT [0:1<<TOTAL_BITS]; // Activation LUT with 256 entries storing 8-bit data

    // Load the LUT from the memory file if specified
    initial begin
        if (ACT_FUNC_MEM != "") begin
            $display("Loading LUT from %s", ACT_FUNC_MEM);
            $readmemh(ACT_FUNC_MEM, activation_LUT);
        end
    end

    logic signed [TOTAL_BITS*2-1:0] wx [0:(INPUT_CNT-1)];
    logic signed [TOTAL_BITS*2-1:0] sum;
    logic signed [TOTAL_BITS-1:0] roundedSum;

    always @(posedge clk) begin
        // Initialize the sum with bias
        sum = bias << FRAC_BITS;

        // Generate loop to handle the multiplication and addition for any INPUT_CNT
        for (int i = 0; i < INPUT_CNT; i++) begin
            wx[i] = w[i] * x[i];
            sum += wx[i];
        end

        roundedSum = {sum[(TOTAL_BITS*2)-1], sum[FRAC_BITS+FRAC_BITS+INT_BITS-2:FRAC_BITS]};

        // // Check for fixed point overflow during calculations
        // if ((sum < 0 && bias < 0 && roundedSum > 0) || (sum > 0 && bias > 0 && roundedSum < 0)) begin
        //     if (sum < 0) begin
        //         roundedSum = MIN_VAL;
        //     end else begin
        //         roundedSum = MAX_VAL;
        //     end
        // end

        // Check for out-of-bounds error from the rounding
        for (int i = (TOTAL_BITS*2)-2; i >= FRAC_BITS+FRAC_BITS+INT_BITS-1; i--) begin
            if (sum[i] != sum[TOTAL_BITS*2-1]) begin
                if (sum[TOTAL_BITS*2-1] == 1) begin
                    roundedSum = MIN_VAL;

                end else begin
                    roundedSum = MAX_VAL;
                end
                break;
            end
        end

        // Use the activation LUT if specified, otherwise use the rounded sum directly
        if (ACT_FUNC_MEM != "") begin
            y <= activation_LUT[roundedSum + (1<<(TOTAL_BITS-1))];
        end else begin
            y <= roundedSum; // Use sum directly if not using LUT
        end
    end

endmodule
