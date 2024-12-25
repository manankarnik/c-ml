#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    float *input;
    float *weight;
    float bias;
    size_t size;
} Perceptron;

float randf()
{
    return (float) rand() / (float) RAND_MAX;
}

void init_perceptron(Perceptron *p, float *input, size_t size)
{
    p->weight = malloc(size);
    for (size_t i = 0; i < size; ++i) {
        p->weight[i] = randf();
    }
    p->input = input;
    p->bias = 1;
    p->size = size;
}

void free_perceptron(Perceptron *p)
{
    if (p->weight != NULL) {
        free(p->weight);
        p->weight = NULL;
    }
}

int main(void)
{
    srand(time(0));
    float input[] = {1, 1, 1};
    size_t size = sizeof(input) / sizeof(input[0]);

    Perceptron p;
    init_perceptron(&p, input, size);

    printf("inputs = ");
    for (size_t i = 0; i < size; ++i) {
        printf("%f ", p.input[i]);
    }
    printf("\nweights = ");
    for (size_t i = 0; i < size; ++i) {
        printf("%f ", p.weight[i]);
    }
    printf("\nbias = %f\n", p.bias);

    free_perceptron(&p);
    return 0;
}
