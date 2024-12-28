#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <assert.h>

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

float loss(Perceptron p, float *ti, float *to, size_t output_size)
{
    float result = 0;
    float input[p.input_size];
    for (size_t i = 0; i < output_size; ++i) {
        for (size_t j = 0; j < p.input_size; ++j) {
            input[j] = ti[i*p.input_size + j];
        }
        float prediction = predict_perceptron(p, input);
        float error = prediction - to[i];
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

void train_perceptron(float eps, Perceptron *p, float *ti, float* to, size_t output_size, float learning_rate, size_t epochs)
{
    for (size_t i = 0; i < epochs; ++i) {
        float l = loss(*p, ti, to, output_size);
        for (size_t j = 0; j < p->input_size; ++j) {
            float old_weight = p->weight[j];
            p->weight[j] += eps;
            float delta = (loss(*p, ti, to, output_size) - l)/eps;
            p->weight[j] = old_weight;
            p->weight[j] -= learning_rate*delta;
        }
        float old_bias = p->bias;
        p->bias += eps;
        float delta = (loss(*p, ti, to, output_size) - l)/eps;
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

void csv_to_array(char *buffer, size_t buffer_size, float *ip, size_t *ip_size, float *op, size_t *op_size)
{
    size_t i = 0;
    *op_size = 0;
    *ip_size = 0;
    size_t line_size = strcspn(buffer, "\n");
    while (i < buffer_size) {
        if (i%(line_size + 1) != (line_size - 1)) {
            ip[*ip_size] = (float) atoi(buffer);
            (*ip_size)++;
        } else {
            op[*op_size] = (float) atoi(buffer);
            (*op_size)++;
        }
        i += 2;
        buffer += 2;
    }
}

int main(int argc, char **argv)
{
    if (argc == 1) {
        fprintf(stderr, "ERROR: Train file required\n");
        return 1;
    }
    else if (argc == 2) {
        fprintf(stderr, "ERROR: Predict file required\n");
        return 1;
    }

    FILE *tf = fopen(argv[1], "r");
    if (tf == NULL) {
        fprintf(stderr, "ERROR: %s - %s\n", argv[1], strerror(errno));
        return 1;
    }
    fseek(tf, 0, SEEK_END);
    size_t tf_size = ftell(tf);
    fseek(tf, 0, SEEK_SET);
    char tf_buf[tf_size];
    fread(tf_buf, sizeof(*tf_buf), sizeof(tf_buf)/sizeof(tf_buf[0]), tf);
    fclose(tf);
    float *ti = malloc(tf_size*sizeof(float));
    float *to = malloc(tf_size*sizeof(float));
    size_t ti_size, to_size;
    csv_to_array(tf_buf, tf_size, ti, &ti_size, to, &to_size);

    FILE *pf = fopen(argv[2], "r");
    if (pf == NULL) {
        fprintf(stderr, "ERROR: %s - %s\n", argv[2], strerror(errno));
        return 1;
    }
    fseek(pf, 0, SEEK_END);
    size_t pf_size = ftell(pf);
    fseek(pf, 0, SEEK_SET);
    char pf_buf[pf_size];
    fread(pf_buf, sizeof(*pf_buf), sizeof(pf_buf)/sizeof(pf_buf[0]), pf);
    fclose(pf);
    float *pi = malloc(pf_size*sizeof(float));
    float *po = malloc(pf_size*sizeof(float));
    size_t pi_size, po_size;
    csv_to_array(pf_buf, pf_size, pi, &pi_size, po, &po_size);

    assert(ti_size/to_size == pi_size/po_size);

    size_t input_size = ti_size/to_size;
    srand(time(0));
    float learning_rate = 1e-1;
    float eps = 1e-1;
    size_t epochs = 1000;

    Perceptron p;
    init_perceptron(&p, input_size);
    printf("\nINFO: Perceptron initialized!\n");
    print_perceptron(p);
    printf("\nINFO: Training perceptron...\n");
    train_perceptron(eps, &p, ti, to, to_size, learning_rate, epochs);
    printf("\nINFO: Training completed!\n");
    print_perceptron(p);

    printf("\nINFO: Predicting...");
    float input[p.input_size];
    for (size_t i = 0; i < po_size; ++i) {
        printf("\n");
        for (size_t j = 0; j < p.input_size; ++j) {
            input[j] = pi[i*p.input_size + j];
        }
        print_array("inputs", input, p.input_size);
        float prediction = predict_perceptron(p, input);
        printf("output = %f\nprediction = %f\n", po[i], prediction);
    }

    free_perceptron(&p);
    free(ti);
    free(to);
    free(pi);
    free(po);
    return 0;
}
