open_project -reset proj_mlp_optimized
set_top MLP_Forward_Pass
add_files mlp_optimized.cpp -cflags "-std=c++11"
add_files weights_small_other.cpp -cflags "-std=c++11"
add_files mlp.h
add_files -tb verification_tb.cpp -cflags "-std=c++11"
add_files -tb C:/Users/jvald/HLS_Export/w1.txt
add_files -tb C:/Users/jvald/HLS_Export/mnist_test.csv
open_solution -reset "solution1" -flow_target vivado
set_part {xcku5p-ffvb676-2-e}
create_clock -period 10 -name default
config_compile -pipeline_loops 0
csim_design
close_project
exit
