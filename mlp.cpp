#include "mlp.h"
#include "weights_small.h"

// Use BRAM for the large weight matrix W1 and other weights
#pragma HLS BIND_STORAGE variable=W1 type=rom_np impl=bram
#pragma HLS ARRAY_RESHAPE variable=W1 cyclic factor=4 dim=2
#pragma HLS BIND_STORAGE variable=b1 type=rom_np impl=bram
#pragma HLS BIND_STORAGE variable=W2 type=rom_np impl=bram
#pragma HLS BIND_STORAGE variable=b2 type=rom_np impl=bram

bool MLP_Forward_Pass (
    float input_vector[N_INPUTS]
) {
    // #pragma HLS INTERFACE ...

    // Buffer for hidden layer
    float encoded_vector[N_HIDDEN];
    // Use URAM for the large hidden layer buffer
    #pragma HLS BIND_STORAGE variable=encoded_vector type=ram_2p impl=uram


    // Layer 1: W1 * input + b1 + ReLU
    for (int i = 0; i < N_HIDDEN; i++) {
        float sum = (float)b1[i];
        for (int j = 0; j < N_INPUTS; j++) {
            sum += (float)W1[i][j] * input_vector[j];
        }
        encoded_vector[i] = (sum < 0.0f) ? 0.0f : sum; // ReLU
    }

    // Layer 2: W2 * encoded + b2
    // Local output vector since we removed the argument
    float output_vector[N_OUTPUTS];
    float sum_exp = 0.0f;
    for (int i = 0; i < N_OUTPUTS; i++) {
        float sum = (float)b2[i];
        for (int j = 0; j < N_HIDDEN; j++) {
            sum += (float)W2[i][j] * encoded_vector[j];
        }
        output_vector[i] = sum; // Temporary store before exp
    }

    // Softmax Logic & Argmax
    int max_idx = 0;
    float max_val = -1.0f; 

    for (int i = 0; i < N_OUTPUTS; i++) {
        output_vector[i] = exp(output_vector[i]);
        sum_exp += output_vector[i];
    }
    for (int i = 0; i < N_OUTPUTS; i++) {
        output_vector[i] /= sum_exp;
        if (output_vector[i] > max_val) {
            max_val = output_vector[i];
            max_idx = i;
        }
    }
    
    // Return true if class 1 (Abnormality), false if class 0
    return (max_idx == 1);
}
