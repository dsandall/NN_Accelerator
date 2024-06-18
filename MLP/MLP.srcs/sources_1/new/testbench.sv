`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/27/2024 12:20:12 PM
// Design Name: 
// Module Name: testbench
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


module testbench(
    );
    
    reg CLK = 0;
    reg signed [BITS_INPUT-1:0] input_x;
    reg signed [BITS_OUTPUT-1:0] output_y;

    topWrapper wrapperInstance (.CLK(CLK), .input_x(input_x), .output_y(output_y));

    initial forever  #5  CLK =  !CLK; 

    integer file;
    initial begin
      // Open file for writing
      file = $fopen("Neuron_Test1_Output.txt", "w");
      if (!file) begin
          $display("Error: Could not open file.");
          $finish;
      end

      input_x = 0;

    end
   
    initial forever begin
      input_x = input_x + 1;
      //btnc=1;

      #20;
      $monitor("Time: %0t, input: %h, output: %h", $time, input_x, output_y);
      $fdisplay(file, "Time: %0t, input: %h, output: %h", $time, input_x, output_y);


      //$finish;
      if (input_x == 320) begin
        $fclose(file);
        $finish;
      end
    end


    // // Monitor output
    // function real q46_to_real(input signed [9:0] q46_value);
    //     q46_to_real = q46_value / 2.0 ** 6;
    // endfunction


    
endmodule
