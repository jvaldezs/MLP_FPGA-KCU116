#include "mlp.h"
#include "weights_small.h"

bool MLP_Forward_Pass (
    float input_vector[N_INPUTS],
    bool load_weights,
    weight_t weights_ext[N_HIDDEN * N_INPUTS]
) {
    #pragma HLS INTERFACE m_axi port=weights_ext offset=slave bundle=gmem_weights depth=1800000

    // Weight Storage Pragmas
    static weight_t W1_URAM[N_URAM_HIDDEN][N_INPUTS];
    #pragma HLS BIND_STORAGE variable=W1_URAM type=ram_2p impl=uram
    #pragma HLS ARRAY_RESHAPE variable=W1_URAM cyclic factor=4 dim=2

    static weight_t W1_BRAM[N_BRAM_HIDDEN][N_INPUTS];
    #pragma HLS BIND_STORAGE variable=W1_BRAM type=ram_2p impl=bram
    #pragma HLS ARRAY_RESHAPE variable=W1_BRAM cyclic factor=4 dim=2

    #pragma HLS ARRAY_PARTITION variable=b1 cyclic factor=16 dim=1
    #pragma HLS BIND_STORAGE variable=W2 type=ram_2p impl=lutram
    #pragma HLS BIND_STORAGE variable=b2 type=ram_2p impl=lutram

    // One-time initialization from external memory
    if (load_weights) {
        Load_W1_URAM: for (int i = 0; i < N_URAM_HIDDEN; i++) {
            for (int j = 0; j < N_INPUTS; j++) {
                #pragma HLS PIPELINE II=1
                W1_URAM[i][j] = weights_ext[i * N_INPUTS + j];
            }
        }
        Load_W1_BRAM: for (int i = 0; i < N_BRAM_HIDDEN; i++) {
            for (int j = 0; j < N_INPUTS; j++) {
                #pragma HLS PIPELINE II=1
                W1_BRAM[i][j] = weights_ext[(i + N_URAM_HIDDEN) * N_INPUTS + j];
            }
        }
        return false;
    }

    // Hidden layer buffer
    float encoded_vector[N_HIDDEN];
    #pragma HLS BIND_STORAGE variable=encoded_vector type=ram_2p impl=uram

    // Layer 1: Hybrid Dot Product (URAM + BRAM)
    
    // Pass 1: URAM Neurons (0 to 3399)
    L1_URAM_Neurons: for (int i = 0; i < N_URAM_HIDDEN; i++) {
        float sum = (float)b1[i];
        L1_URAM_DotProduct: for (int j = 0; j < N_INPUTS; j++) {
            #pragma HLS PIPELINE II=1
            sum += (float)W1_URAM[i][j] * input_vector[j];
        }
        encoded_vector[i] = (sum < 0.0f) ? 0.0f : sum; 
    }

    // Pass 2: BRAM Neurons (3400 to 5999)
    L1_BRAM_Neurons: for (int i = 0; i < N_BRAM_HIDDEN; i++) {
        float sum = (float)b1[i + N_URAM_HIDDEN];
        L1_BRAM_DotProduct: for (int j = 0; j < N_INPUTS; j++) {
            #pragma HLS PIPELINE II=1
            sum += (float)W1_BRAM[i][j] * input_vector[j];
        }
        encoded_vector[i + N_URAM_HIDDEN] = (sum < 0.0f) ? 0.0f : sum; 
    }

    // Layer 2: W2 * encoded + b2
    float output_vector[N_OUTPUTS];
    float sum_exp = 0.0f;
    L2_Neurons: for (int i = 0; i < N_OUTPUTS; i++) {
        float sum = (float)b2[i];
        
        L2_DotProduct: for (int j = 0; j < N_HIDDEN; j++) {
            #pragma HLS PIPELINE II=1 
            sum += (float)W2[i][j] * encoded_vector[j];
        }
        output_vector[i] = sum; 
    }

    // Softmax Logic
    Softmax_Exp: for (int i = 0; i < N_OUTPUTS; i++) {
        #pragma HLS PIPELINE II=1
        output_vector[i] = exp(output_vector[i]);
        sum_exp += output_vector[i];
    }
    
    int max_idx = 0;
    float max_val = -1.0f;

    Softmax_Norm: for (int i = 0; i < N_OUTPUTS; i++) {
        #pragma HLS PIPELINE II=1
        float val = output_vector[i] / sum_exp;
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    return (max_idx == 1);
}
