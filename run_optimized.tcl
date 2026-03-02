# Vitis HLS Tcl script for OPTIMIZED MLP design (Mod0)
# This script synthesizes the pragma-optimized design

# Create new project
open_project -reset proj_mlp_optimized

# Set top-level function
set_top MLP_Forward_Pass

# Add design files
add_files mlp_optimized.cpp -cflags "-std=c++11"
add_files weights_small_other.cpp -cflags "-std=c++11"
# add_files weights_small_bram.cpp -cflags "-std=c++11"

add_files mlp.h

# Add testbench files
add_files -tb verification_tb.cpp -cflags "-std=c++11"
add_files -tb w1.txt
add_files -tb mnist_test.csv 

# Create solution
open_solution -reset "solution1" -flow_target vivado

# Set target device - modify according to your FPGA board
set_part {xcku5p-ffvb676-2-e}

# Set clock period (10ns = 100MHz)
create_clock -period 10 -name default

# Configure synthesis
config_compile -pipeline_loops 0

# Run C simulation
csim_design

# Run synthesis
csynth_design

# Close project
close_project

exit
