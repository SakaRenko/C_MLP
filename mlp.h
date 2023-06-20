#ifndef MLP_H_INCLUDED
#define MLP_H_INCLUDED

#include "functions.h"
#include "vector"

class MLP {
public:
    MLP(int input_dim, int hidden_dim, int output_dim);

    std::vector<double> forward(const std::vector<unsigned char> &, int tn);

    void zero_grad(int tn);

    void backward(const std::vector<double> &y, const std::vector<double> &y_hat, int tn);

    void update(double lr, int tn);

    ~MLP();

private:
    static const int NUM_THREAD = 4;

    std::vector<std::vector<double>> W1;
    std::vector<std::vector<double>> W2;
    std::vector<double> b1;
    std::vector<double> b2;
    std::vector<std::vector<double>> W1_grad[NUM_THREAD];
    std::vector<std::vector<double>> W2_grad[NUM_THREAD];
    std::vector<double> b1_grad[NUM_THREAD];
    std::vector<double> b2_grad[NUM_THREAD];

    vector<double> input[NUM_THREAD];
    vector<double> y1[NUM_THREAD];
    vector<double> z1[NUM_THREAD];
};


#endif // MLP_H_INCLUDED
