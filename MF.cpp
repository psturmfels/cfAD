#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <vector>
#include <algorithm>
#include <iterator>
#include <random>
#include <iostream>
#include <numeric>
#include <functional>

namespace p = boost::python;
namespace np = boost::python::numpy;

float dot_product(float const * const v1, float const * const v2, const int length) 
{
    return std::inner_product(v1, (v1 + length), v2, 0.0);
}

void update_gradients(float * const u_i_ptr, float * const v_j_ptr, float e_ij, float eta, float lamb1, int k, std::vector<float> neighbor_average, float lamb2) 
{
    for (int l = 0; l < k; ++l) 
    {
        float u_il = *(u_i_ptr + l);
        float v_jl = *(v_j_ptr + l);
        
        *(u_i_ptr + l) += eta * (e_ij * v_jl - lamb1 * u_il);
        *(v_j_ptr + l) += eta * (e_ij * u_il - lamb1 * v_jl + lamb2 * neighbor_average[l]);
    }
}

void update_gradients(float * const u_i_ptr, float * const v_j_ptr, float e_ij, float eta, float lamb1, int k) 
{
    for (int l = 0; l < k; ++l) 
    {
        float u_il = *(u_i_ptr + l);
        float v_jl = *(v_j_ptr + l);
        
        *(u_i_ptr + l) += eta * (e_ij * v_jl - lamb1 * u_il);
        *(v_j_ptr + l) += eta * (e_ij * u_il - lamb1 * v_jl);
    }
}

float Get_prediction_error(np::ndarray X, np::ndarray U, np::ndarray V, np::ndarray known_indices) 
{
    int k = U.shape(1);
    int numKnown = known_indices.shape(0);
    
    float error = 0.0;
    for (int known_ind = 0; known_ind < numKnown; ++known_ind) 
    { 
        int i = p::extract<int>(known_indices[known_ind][0]);
        int j = p::extract<int>(known_indices[known_ind][1]);

        float x_ij = p::extract<float>(X[i][j]);
        np::ndarray u_i = p::extract<np::ndarray>(U[i]);
        np::ndarray v_j = p::extract<np::ndarray>(V[j]);

        float * u_i_ptr = reinterpret_cast<float *>(u_i.get_data());
        float * v_j_ptr = reinterpret_cast<float *>(v_j.get_data());

        float e_ij = x_ij - dot_product(u_i_ptr, v_j_ptr, k);
        
        error += e_ij * e_ij;
    }
    error = error / numKnown;
    
    return error;
}

//note to self: we need to work with 32-bit floats and integers to make c++ happy.
void Factor(np::ndarray X, np::ndarray U, np::ndarray V, 
            np::ndarray train_indices, np::ndarray test_indices, p::dict neighbors, 
            np::ndarray train_errors, np::ndarray test_errors, bool tabulate_errors=false,
            float eta=0.005, float lamb1=0.02, float lamb2=0.0001, int num_epochs=200)
{
    int numKnown = train_indices.shape(0);
    std::random_device rd;
    std::mt19937 g(rd());
    
    int k = U.shape(1);
    float best_error = 0.0;
    int epochs_since_last_best = 0;
    int early_stopping_epochs = 1;
    
    float * train_error_ptr = NULL;
    float * test_error_ptr  = NULL;
    if (tabulate_errors) 
    {
        train_error_ptr = reinterpret_cast<float *>(train_errors.get_data());
        test_error_ptr  = reinterpret_cast<float *>(test_errors.get_data());
    }
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) 
    {
        std::vector<int> v(numKnown);
        std::iota(v.begin(), v.end(), 0);
        std::shuffle(v.begin(), v.end(), g);

        float average_epoch_error = 0.0;
        for (int known_ind = 0; known_ind < numKnown; ++known_ind) 
        {
            int rand_ind = v[known_ind];

            int i = p::extract<int>(train_indices[rand_ind][0]);
            int j = p::extract<int>(train_indices[rand_ind][1]);

            float x_ij = p::extract<float>(X[i][j]);
            np::ndarray u_i = p::extract<np::ndarray>(U[i]);
            np::ndarray v_j = p::extract<np::ndarray>(V[j]);

            float * u_i_ptr = reinterpret_cast<float *>(u_i.get_data());
            float * v_j_ptr = reinterpret_cast<float *>(v_j.get_data());

            float e_ij = x_ij - dot_product(u_i_ptr, v_j_ptr, k);
            average_epoch_error += e_ij * e_ij;

            //Calculate average of neighbors if there are any
            if (neighbors.has_key(j))
            {
                p::list n_j = p::extract<p::list>(neighbors[j]);
                int num_neighbors = p::len(n_j);

                std::vector<float> neighbor_average(k, 0.0);
                for (int n_ind = 0; n_ind < num_neighbors; ++n_ind) 
                {
                    p::list neighbor_list = p::extract<p::list>(n_j[n_ind]);
                    int neighbor = p::extract<int>(neighbor_list[0]);
                    float neighbor_weight = p::extract<float>(neighbor_list[1]);
                    
                    np::ndarray neighbor_vec = p::extract<np::ndarray>(V[neighbor]);
                    float * neighbor_ptr = reinterpret_cast<float *>(neighbor_vec.get_data());

                    for (int l = 0; l < k; ++l) 
                    {
                        neighbor_average[l] += *(neighbor_ptr + l) * neighbor_weight;
                    }
                }

                update_gradients(u_i_ptr, v_j_ptr, e_ij, eta, lamb1, k, neighbor_average, lamb2);
            }
            else 
            {
                update_gradients(u_i_ptr, v_j_ptr, e_ij, eta, lamb1, k);  
            }
        }
        
        average_epoch_error = average_epoch_error / numKnown;
        float current_test_error = Get_prediction_error(X, U, V, test_indices);
        
        if (tabulate_errors)
        {
            std::cout << "Epoch: " << epoch << ", train: " << average_epoch_error << ", test: " << current_test_error << std::endl;
            *(train_error_ptr + epoch) = average_epoch_error;
            *(test_error_ptr  + epoch) = current_test_error;
        }
        
        if (best_error > current_test_error || epoch == 0) 
        {
            best_error = current_test_error;
            epochs_since_last_best = 0;
        }
        else if (best_error < current_test_error)
        {
            epochs_since_last_best += 1;
            if (epochs_since_last_best > early_stopping_epochs)
            {
                break;
            }
        }
    }
    return;
}

BOOST_PYTHON_MODULE(MF)
{
    np::initialize();
    p::def("Factor", Factor);
    p::def("Get_prediction_error", Get_prediction_error);
}