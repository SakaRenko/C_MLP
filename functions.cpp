#include "functions.h"
#include <cmath>
#include <cassert>
#include <immintrin.h>
#include<iostream>

using namespace std;

double cross_entropy(const vector<double> &y, const vector<double> &y_hat) {
    double loss = 0;
    for (int i = 0; i < y.size(); i++) {
        loss += -y[i] * log(y_hat[i]);
    }
    return loss;
}

vector<double> sigmoid(vector<double> x) {
    for (auto &i: x) {
        i = 1 / (1 + exp(-i));
    }
    return x;
}

vector<double> d_sigmoid(const vector<double> &x) {
    auto sigmoid_x = sigmoid(x);
    auto d = vector<double>(sigmoid_x.size(), 0);
    for (int i = 0; i < sigmoid_x.size(); i++) {
        d[i] = sigmoid_x[i] * (1 - sigmoid_x[i]);
    }
    return d;
}


vector<double> softmax(const vector<double> &x) {
    auto ex = vector<double>(x.size(), 0);
    double sum = 0;
    for (int i = 0; i < x.size(); i++) {
        ex[i] = exp(x[i]);
        sum += ex[i];
    }
    for (int i = 0; i < x.size(); i++) {
        ex[i] /= sum;
    }
    return ex;
}

vector<double> d_softmax_cross_entropy(const vector<double> &y, const vector<double> &y_hat) {
    auto d = vector<double>(y.size(), 0);
    for (int i = 0; i < y.size(); i++) {
        d[i] = y_hat[i] - y[i];
    }
    return d;
}

//vector<vector<double>> outer_product(const vector<double> &x, const vector<double> &y) {
//    auto result = vector<vector<double>>(x.size(), vector<double>(y.size(), 0));
//    for (int i = 0; i < x.size(); i++) {
//        for (int j = 0; j < y.size(); j++) {
//            result[i][j] = x[i] * y[j];
//        }
//    }
//    return result;
//}

vector<vector<double>> outer_product(const vector<double> &x, const vector<double> &y) {
    auto result = vector<vector<double>>(x.size(), vector<double>(y.size(), 0));
    __m256d t2, t3;
    for (int i = 0; i < x.size(); i++) {
        int j = 0;
        double x_m[4] = {x[i], x[i], x[i], x[i]};
        t2 = _mm256_loadu_pd(x_m);
        const double * b_pointer = &(y[0]);
        double * r_pointer = &(result[i][0]);
        for (j; j <= y.size() - 4; j+=4) {
            t3 = _mm256_loadu_pd(b_pointer + j);
            t3 = _mm256_mul_pd(t2, t3);
            _mm256_storeu_pd(r_pointer + j, t3);
        }
        for (j; j < y.size(); j++) {
            result[i][j] += x[i] * y[j];
        }
    }
    return result;
}

void matrix_add(vector<vector<double>> &x, const vector<vector<double>> &y) {
    assert(x.size() == y.size());
    assert(x[0].size() == y[0].size());
    for (int i = 0; i < x.size(); i++) {
        for (int j = 0; j < x[0].size(); j++) {
            x[i][j] += y[i][j];
        }
    }
}

void vector_add(vector<double> &x, const vector<double> &y) {
    assert(x.size() == y.size());
    for (int i = 0; i < x.size(); i++) {
        x[i] += y[i];
    }
}

void matrix_multiply(std::vector<std::vector<double>> &x, double y) {
    for (int i = 0; i < x.size(); i++) {
        for (int j = 0; j < x[0].size(); j++) {
            x[i][j] *= y;
        }
    }
}

//void matrix_multiply(std::vector<std::vector<double>> &x, double y) {
//    __m256d t1, t2, t3;
//    double y_t[4] = {y, y, y, y};
//    t1 = _mm256_loadu_pd(y_t);
//    for (int i = 0; i < x.size(); i++) {
//        int j = 0;
//        double * a_pointer = &(x[i][0]);
//        for (j; j <= x[0].size() - 4; j+=4) {
//            t3 = _mm256_loadu_pd(a_pointer + j);
//            t3 = _mm256_mul_pd(t1, t3);
//            _mm256_storeu_pd(a_pointer + j, t3);
//        }
//        for (j; j < x[0].size(); j++) {
//            x[i][j] *= y;
//        }
//    }
//}

void vector_multiply(std::vector<double> &x, double y) {
    for (double & i : x) {
        i *= y;
    }
}

double random(double min, double max) {
    return min + (max - min) * rand() / (RAND_MAX + 1.0);
}


vector<double> matrix_dot(vector<vector<double>> a, vector<double> b) {
    vector<double> result;
    __m256d t1, t2, t3;
    for (auto & i : a) {
        double sum[4] = {0};
        t1 = _mm256_loadu_pd(sum);
        double sum_all = 0;
        double * a_pointer = &(i[0]);
        double * b_pointer = &(b[0]);
        int j = 0;
        for (j; j < i.size() - 4; j+=4) {
            t2 = _mm256_loadu_pd(a_pointer + j);
            //cout<<*(a_pointer + j)<<" "<<i[j]<<endl;
            t3 = _mm256_loadu_pd(b_pointer + j);
            t3 = _mm256_mul_pd(t2, t3);
            t1 = _mm256_add_pd(t1, t3);
        }
        for(j; j < i.size(); j++)
            sum_all += i[j] * b[j];
        _mm256_storeu_pd(sum, t1);
        sum_all += sum[0] + sum[1] + sum[2] + sum[3];
        result.push_back(sum_all);
    }
    return result;
}

//vector<double> matrix_dot(vector<vector<double>> a, vector<double> b) {
//    vector<double> result;
//    for (auto & i : a) {
//        double sum = 0;
//        for (int j = 0; j < i.size(); j++) {
//            sum += i[j] * b[j];
//        }
//        result.push_back(sum);
//    }
//    return result;
//}


vector<double> vector_dot(vector<double>x, vector<double>y){
    assert(x.size() == y.size());
    vector<double> result;
    result.reserve(x.size());
    for(int i = 0;i<x.size();i++){
        result.push_back(x[i]*y[i]);
    }
    return result;
}

vector<vector<double>> transpose(vector<vector<double>> x) {
    int m = x.size(), n = x[0].size();
    vector<vector<double>> result(n, vector<double>(m));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[j][i] = x[i][j];
        }
    }
    return result;
}
