#include <unistd.h>
#include <iostream>
#include <string>
#include <getopt.h>
#include <cassert>
#include "utils.h"
#include "mnist_reader_less.h"
#include "mlp.h"
#include <immintrin.h>
#include <pthread.h>
#include<Windows.h>

using namespace std;

const int NUM_THREADS = 4;

struct param_mlp {
        double lr;
        int epoch_num;
        int hidden_dim;
        int r;
        string dataset_path;
        MLP *mlp;
};

pthread_barrier_t barrier;
pthread_mutex_t mutex;


void* train(void* param) {
    struct param_mlp *p = (struct param_mlp *)param;
    double learning_rate = p->lr;
    int epoch_num = p->epoch_num;
    int hidden_dim = p->hidden_dim;
    int r = p->r;
    string dataset_path = p->dataset_path;
    printf("Learning rate: %f, epoch number: %d, hidden dimension: %d, dataset path: %s\n", learning_rate, epoch_num, hidden_dim, dataset_path.c_str());
    // Read the MNIST dataset
    auto training_images = mnist::read_mnist_image_file<uint8_t>(dataset_path + "/train-images-idx3-ubyte");
    auto training_labels = mnist::read_mnist_label_file<uint8_t>(dataset_path + "/train-labels-idx1-ubyte");
    auto test_images = mnist::read_mnist_image_file<uint8_t>(dataset_path + "/t10k-images-idx3-ubyte");
    auto test_labels = mnist::read_mnist_label_file<uint8_t>(dataset_path + "/t10k-labels-idx1-ubyte");
    printf("Training images: %zu x %zu\n", training_images.size(), training_images[0].size());
    printf("Training labels: %zu\n", training_labels.size());
    assert(training_images.size() == training_labels.size());
    printf("Test images: %zu x %zu\n", test_images.size(), test_images[0].size());
    printf("Test labels: %zu\n", test_labels.size());
    assert(test_images.size() == test_labels.size());


    // Create a neural network with 784 inputs, 100 hidden neurons and 10 outputs
    MLP* mlp = (p->mlp);

    srand(time(NULL));
    LARGE_INTEGER timeStart;
    LARGE_INTEGER timeEnd;

    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    double quadpart = (double)frequency.QuadPart;

    int training_real = training_images.size() - training_images.size() % NUM_THREADS;

    // Train the network
    for (int epoch = 0; epoch < epoch_num; epoch++) {
        vector<double> losses;
        QueryPerformanceCounter(&timeStart);
        for (int i = 0; i <= training_real - NUM_THREADS; i += NUM_THREADS) {
            auto x = training_images[i + r];
            auto l = training_labels[i + r];
            vector<double> y(10, 0);
            y[l] = 1;
            auto y_hat = mlp->forward(x, r);
            auto loss = cross_entropy(y, y_hat);
            losses.push_back(loss);
            if (i % 1000 == 0 && r == 0) {
                double sum = 0;
                QueryPerformanceCounter(&timeEnd);
                double _time = (timeEnd.QuadPart - timeStart.QuadPart) / quadpart;
                for (auto &l: losses) {
                    sum += l;
                }
                double avg_loss = sum / losses.size();
                losses.clear();
                printf("Epoch: %d, Iteration: %d, Loss: %f\n, Time: %f\n", epoch, i, avg_loss, _time);
                QueryPerformanceCounter(&timeStart);
            }
            pthread_barrier_wait(&barrier);
            mlp->zero_grad(r);
            mlp->backward(y, y_hat, r);
            pthread_mutex_lock(&mutex);
            mlp->update(learning_rate, r);
            pthread_mutex_unlock(&mutex);
            pthread_barrier_wait(&barrier);
        }
    }
    pthread_exit(nullptr);
}


int main(int argc, char *argv[]) {
    pthread_barrier_init(&barrier,NULL,NUM_THREADS);
    pthread_mutex_init(&mutex, NULL);
     pthread_t thread[NUM_THREADS];
    struct param_mlp thread_param[NUM_THREADS];
    MLP mlp(784,100, 10);
    for(int i=0;i<NUM_THREADS;i++){
        thread_param[i].lr = 0.001;
        thread_param[i].epoch_num = 10;
        thread_param[i].r = i;
        thread_param[i].dataset_path = "D:\\c++homework\\parallel programing\\NN\\neural_network\\data\\";
        thread_param[i].hidden_dim = 100;
        thread_param[i].mlp = &mlp;
    }
//    double learning_rate = 0.001;
//    int epoch_num = 10;
//    string dataset_path = "D:\\c++homework\\parallel programing\\NN\\neural_network\\data\\";
//    int hidden_dim = 100;
//    train(learning_rate, epoch_num, hidden_dim, dataset_path);
    for(int i=0;i<NUM_THREADS;i++){
        pthread_create(&thread[i],nullptr,train,(void*)(&thread_param[i]));
    }
    for(int i=0;i<NUM_THREADS;i++){
        pthread_join(thread[i],nullptr);
    }
    return 0;
}
