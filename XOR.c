#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_NODES 2
#define HIDDEN_NODES 4
#define OUTPUT_NODES 1
#define LEARNING_RATE 0.7

typedef struct {
    double input[INPUT_NODES];
    double hidden[HIDDEN_NODES];
    double output[OUTPUT_NODES];
} Layer;

typedef struct {
    double input_hidden[INPUT_NODES][HIDDEN_NODES];
    double hidden_output[HIDDEN_NODES][OUTPUT_NODES];
    double hidden_bias[HIDDEN_NODES];
    double output_bias[OUTPUT_NODES];
} Weights;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

void initialize_weights(Weights *weights) {
    srand(time(0));
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            weights->input_hidden[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }

    for (int i = 0; i < HIDDEN_NODES; i++) {
        for (int j = 0; j < OUTPUT_NODES; j++) {
            weights->hidden_output[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
        weights->hidden_bias[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    for (int i = 0; i < OUTPUT_NODES; i++) {
        weights->output_bias[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

void forward_propagation(Layer *layer, Weights *weights) {
    for (int i = 0; i < HIDDEN_NODES; i++) {
        layer->hidden[i] = weights->hidden_bias[i];
        for (int j = 0; j < INPUT_NODES; j++) {
            layer->hidden[i] += layer->input[j] * weights->input_hidden[j][i];
        }
        layer->hidden[i] = sigmoid(layer->hidden[i]);
    }

    for (int i = 0; i < OUTPUT_NODES; i++) {
        layer->output[i] = weights->output_bias[i];
        for (int j = 0; j < HIDDEN_NODES; j++) {
            layer->output[i] += layer->hidden[j] * weights->hidden_output[j][i];
        }
        layer->output[i] = sigmoid(layer->output[i]);
    }
}

void backward_propagation(Layer *layer, Weights *weights, double target) {
    double output_errors[OUTPUT_NODES];
    for (int i = 0; i < OUTPUT_NODES; i++) {
        output_errors[i] = target - layer->output[i];
    }

    double hidden_errors[HIDDEN_NODES];
    for (int i = 0; i < HIDDEN_NODES; i++) {
        hidden_errors[i] = 0;
        for (int j = 0; j < OUTPUT_NODES; j++) {
            hidden_errors[i] += output_errors[j] * weights->hidden_output[i][j];
        }
    }

    for (int i = 0; i < HIDDEN_NODES; i++) {
        for (int j = 0; j < OUTPUT_NODES; j++) {
            weights->hidden_output[i][j] += LEARNING_RATE * output_errors[j] * sigmoid_derivative(layer->output[j]) * layer->hidden[i];
        }
    }

    for (int i = 0; i < OUTPUT_NODES; i++) {
        weights->output_bias[i] += LEARNING_RATE * output_errors[i] * sigmoid_derivative(layer->output[i]);
    }

    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            weights->input_hidden[i][j] += LEARNING_RATE * hidden_errors[j] * sigmoid_derivative(layer->hidden[j]) * layer->input[i];
        }
    }

    for (int i = 0; i < HIDDEN_NODES; i++) {
        weights->hidden_bias[i] += LEARNING_RATE * hidden_errors[i] * sigmoid_derivative(layer->hidden[i]);
    }
}

void train(Layer *layer, Weights *weights, double inputs[][INPUT_NODES], double targets[], int epochs, int num_samples) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_error = 0.0;
        for (int sample = 0; sample < num_samples; sample++) {
            for (int i = 0; i < INPUT_NODES; i++) {
                layer->input[i] = inputs[sample][i];
            }
            forward_propagation(layer, weights);
            backward_propagation(layer, weights, targets[sample]);
            
            // Calculate total error for the epoch
            for (int i = 0; i < OUTPUT_NODES; i++) {
                double error = targets[sample] - layer->output[i];
                total_error += error * error;
            }
        }
        printf("Epoch %d, Error: %f\n", epoch + 1, total_error / num_samples);
    }
}

int main() {
    Layer layer;
    Weights weights;
    initialize_weights(&weights);

    double inputs[4][INPUT_NODES] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    double targets[4] = {0, 1, 1, 0};  // XOR

    int epochs = 30000;

    train(&layer, &weights, inputs, targets, epochs, 4);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < INPUT_NODES; j++) {
            layer.input[j] = inputs[i][j];
        }
        forward_propagation(&layer, &weights);
        printf("Input: %d %d, Output: %f\n", (int)inputs[i][0], (int)inputs[i][1], layer.output[0]);
    }

    return 0;
}
