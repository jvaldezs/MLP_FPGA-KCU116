# Vitis HLS Tcl script for ORIGINAL MLP design (baseline)
# This script synthesizes the unoptimized design

# Create new project
open_project -reset proj_mlp_original

# Set top-level function
set_top MLP_Forward_Pass

# Add design files
add_files mlp.cpp -cflags "-std=c++11"
add_files mlp.h

# Add testbench files
add_files -tb verification_tb.cpp -cflags "-std=c++11"
# add_files -tb weightsNew2.txt # File not found in dir listing
add_files -tb mnist_test.csv

# Create solution
open_solution -reset "solution1" -flow_target vivado

# Set target device - modify according to your FPGA board
set_part {xcvu9p-flga2104-2-i}

# Set clock period (10ns = 100MHz)
create_clock -period 10 -name default

# Configure synthesis
config_compile -pipeline_loops 0

# Run C simulation
csim_design

# Run synthesis
csynth_design

# Run co-simulation (optional - comment out if not needed)
# cosim_design -rtl verilog

# Export RTL (optional)
# export_design -format ip_catalog

# Close project
close_project

exit
