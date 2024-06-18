// File: defs.svh
`ifndef DEFS_SVH
`define DEFS_SVH


parameter string HIDDEN_LAYER_MEM = "new_model.mem";


parameter BITS_BIAS = 10; 
parameter BITS_WEIGHT = BITS_BIAS;
parameter BITS_INPUT = BITS_BIAS;
parameter BITS_OUTPUT = BITS_BIAS;
parameter NUM_HIDDEN = 5;
parameter NUM_OUTPUT = 1;

// Define the fixed-point format
parameter int INT_BITS = 4;       // Number of integer bits
parameter int FRAC_BITS = 6;      // Number of fractional bits
parameter int TOTAL_BITS = INT_BITS + FRAC_BITS;

// Masks for integer and fractional parts
parameter int INTEGER_MASK = (1 << INT_BITS) - 1;
parameter int FRACTIONAL_MASK = (1 << FRAC_BITS) - 1;

`endif // DEFS_SVH
