#ifndef MLP_H
#define MLP_H

#include <cmath>
// ----------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------
#include "weights_small.h"
// Using full 32-bit float weights for accuracy

// ----------------------------------------------------------------------
// Function Prototypes
// ----------------------------------------------------------------------
bool MLP_Forward_Pass (
    float input_vector[N_INPUTS],
    bool load_weights,
    weight_t weights_ext[N_HIDDEN * N_INPUTS]
);

#endif // MLP_H
