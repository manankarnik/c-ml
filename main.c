#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <errno.h>

typedef struct {
    float *weight;
    float bias;
    size_t input_size;
} Perceptron;

float randf()
{
    return (float) rand()/(float) RAND_MAX;
}

float sigmoidf(float x)
{
    return 1/(1 + exp(-x));
}

float predict_perceptron(Perceptron p, float *input)
{
    float output = 0;
    for (size_t i = 0; i < p.input_size; ++i) {
        output += input[i]*p.weight[i];
    }
    output += p.bias;
    output = sigmoidf(output);
    return output;
}

float loss(Perceptron p, float *train_input, float *train_output, size_t output_size)
{
    float result = 0;
    float input[p.input_size];
    for (size_t i = 0; i < output_size; ++i) {
        for (size_t j = 0; j < p.input_size; ++j) {
            input[j] = train_input[i*p.input_size + j];
        }
        float prediction = predict_perceptron(p, input);
        float error = prediction - train_output[i];
        result += error*error;
    }
    return result/output_size;
}

void init_perceptron(Perceptron *p, size_t input_size)
{
    p->weight = malloc(input_size);
    for (size_t i = 0; i < input_size; ++i) {
        p->weight[i] = randf();
    }
    p->bias = 1;
    p->input_size = input_size;
}

void train_perceptron(Perceptron *p, float *train_input, float* train_output, size_t output_size, float learning_rate, size_t epochs)
{
    float eps = 1e-3f;
    for (size_t i = 0; i < epochs; ++i) {
        float l = loss(*p, train_input, train_output, output_size);
        for (size_t j = 0; j < p->input_size; ++j) {
            float old_weight = p->weight[j];
            p->weight[j] += eps;
            float delta = (loss(*p, train_input, train_output, output_size) - l)/eps;
            p->weight[j] = old_weight;
            p->weight[j] -= learning_rate*delta;
        }
        float old_bias = p->bias;
        p->bias += eps;
        float delta = (loss(*p, train_input, train_output, output_size) - l)/eps;
        p->bias = old_bias;
        p->bias -= learning_rate*delta;
    }
}

void free_perceptron(Perceptron *p)
{
    if (p->weight != NULL) {
        free(p->weight);
        p->weight = NULL;
    }
}

void print_array(char *label, float *arr, size_t arr_size)
{
    printf("%s = [ ", label);
    for (size_t i = 0; i < arr_size; ++i) {
        printf("%f ", arr[i]);
    }
    printf("]\n");
}

void print_perceptron(Perceptron p)
{
    print_array("weight", p.weight, p.input_size);
    printf("bias = %f\n", p.bias);
}

int main(int argc, char **argv)
{
    if (argc == 1) {
        fprintf(stderr, "ERROR: Training file required\n");
        return 1;
    }

    FILE *f = fopen(argv[1], "r");
    if (f == NULL) {
        fprintf(stderr, "ERROR: %s - %s\n", argv[1], strerror(errno));
        return 1;
    }

    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char buffer[file_size];
    fread(buffer, sizeof(*buffer), sizeof(buffer)/sizeof(buffer[0]), f);
    fclose(f);

    char *b = buffer;
    size_t i = 0;
    size_t train_output_size = 0;
    size_t train_input_size = 0;
    size_t line_size = strcspn(buffer, "\n");
    float *train_input = malloc(file_size*sizeof(float));
    float *train_output = malloc(file_size*sizeof(float));
    while (i < file_size) {
        if (i%(line_size + 1) != (line_size - 1)) {
            train_input[train_input_size] = (float) atoi(b);
            train_input_size++;
        } else {
            train_output[train_output_size] = (float) atoi(b);
            train_output_size++;
        }
        i += 2;
        b += 2;
    }
    print_array("train_input", train_input, train_input_size);
    print_array("train_output", train_output, train_output_size);

    srand(time(0));
    float input[] = {1, 0};
    float learning_rate = 0.8;
    size_t input_size = sizeof(input)/sizeof(input[0]);
    size_t epochs = 100;

    Perceptron p;
    init_perceptron(&p, input_size);
    printf("INFO: Perceptron initialized!\n");
    print_perceptron(p);
    printf("INFO: Training perceptron...\n");
    train_perceptron(&p, train_input, train_output, train_output_size, learning_rate, epochs);
    printf("INFO: Training completed!\n");
    print_array("input", input, input_size);
    print_perceptron(p);
    float output = predict_perceptron(p, input);
    printf("output = %f\n", output);
    free_perceptron(&p);
    free(train_input);
    free(train_output);
    return 0;
}
