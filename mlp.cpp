#include "mlp.h"
using namespace std;

MLP::MLP(int input_dim, int hidden_dim, int output_dim) {
    // Randomly initialize the weights and biases

    W1 = vector<vector<double>>(hidden_dim, vector<double>(input_dim, 0));
    W2 = vector<vector<double>>(output_dim, vector<double>(hidden_dim, 0));
    b1 = vector<double>(hidden_dim, 0);
    b2 = vector<double>(output_dim, 0);
    //initialize W1, W2, b1, b2
    for(int i = 0;i<hidden_dim;i++){
        for (int j = 0; j < input_dim; ++j) {
            W1[i][j] = random(-1,1);
        }
    }
    for(int i = 0;i<output_dim;i++){
        for (int j = 0; j < hidden_dim; ++j) {
            W2[i][j] = random(-1,1);
        }
    }
    for(int i = 0; i<hidden_dim; i++) {
        b1[i] = random(-1,1);
    }
    for (int i = 0; i < output_dim; ++i) {
        b2[i] = random(-1,1);
    }
    // Initialize the gradients
    for(int i = 0; i < NUM_THREAD; i++){
        W1_grad[i] = vector<vector<double>>(W1.size(), vector<double>(W1[0].size(), 0));
        W2_grad[i] = vector<vector<double>>(W2.size(), vector<double>(W2[0].size(), 0));
        b1_grad[i] = vector<double>(b1.size(), 0);
        b2_grad[i] = vector<double>(b2.size(), 0);
    }
}

void MLP::zero_grad(int tn) {
    for (auto &i: W1_grad[tn]) {
        fill(i.begin(), i.end(), 0);
    }
    for (auto &i: W2_grad[tn]) {
        fill(i.begin(), i.end(), 0);
    }
    fill(b1_grad[tn].begin(), b1_grad[tn].end(), 0);
    fill(b2_grad[tn].begin(), b2_grad[tn].end(), 0);
}


vector<double> MLP::forward(const vector<unsigned char> &x, int tn) {
    // implement forward propagation
    input[tn] = vector<double>(x.begin(),x.end());
    y1[tn] = matrix_dot(W1,input[tn]);
    vector_add(y1[tn], b1);
    z1[tn] = sigmoid(y1[tn]);
    vector<double> y2 = matrix_dot(W2,z1[tn]);
    vector_add(y2,b2);
    vector<double> z2 = softmax(y2);
    return z2;
}

void MLP::backward(const vector<double> &y, const vector<double> &y_hat, int tn) {
    b2_grad[tn] = d_softmax_cross_entropy(y,y_hat);
    W2_grad[tn] = outer_product(b2_grad[tn],z1[tn]);
    b1_grad[tn] = vector_dot(matrix_dot(transpose(W2),b2_grad[tn]), d_sigmoid(y1[tn]));
    W1_grad[tn] = outer_product(b1_grad[tn],input[tn]);
}

void MLP::update(double lr, int tn) {
    matrix_multiply(W1_grad[tn],-lr / NUM_THREAD);
    matrix_add(W1, W1_grad[tn]);
    vector_multiply(b1_grad[tn],-lr / NUM_THREAD);
    vector_add(b1,b1_grad[tn]);
    matrix_multiply(W2_grad[tn],-lr / NUM_THREAD);
    matrix_add(W2, W2_grad[tn]);
    vector_multiply(b2_grad[tn],-lr / NUM_THREAD);
    vector_add(b2,b2_grad[tn]);
}

MLP::~MLP() = default;
