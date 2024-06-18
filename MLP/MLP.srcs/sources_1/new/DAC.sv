`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/27/2024 12:15:53 PM
// Design Name: 
// Module Name: ADC
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


module PWM_DAC(
    input logic CLK, 
    input logic [15:0] magnitudeLUT,
    output wire OUT
    );

    reg [3:0] cnt = 0;

    always @(posedge CLK) begin
        if (cnt == 4'b1111) begin cnt = 0; end
        else begin cnt++; end
    end

    assign OUT = magnitudeLUT[cnt];

endmodule


module DS_DAC(
    input logic CLK, 
    input logic RST,
    input logic [N-1:0] magnitude,
    output wire OUT
    );

    parameter N = 16;

    reg [N+1 : 0] sigma;
    reg [N+1 : 0] delta;

    always @(RST) begin
        sigma = 0;
    end       

    always @(posedge CLK) begin
        if (!OUT) begin
            delta = {(N+2){0}};
        end else begin
            delta = {2'b11, {N{1'b0}}};
        end

        sigma = delta + sigma + magnitude;
    end 

    assign OUT = sigma[N+1];


endmodule