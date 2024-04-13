#ifndef ONLINE_CPLEX_H
#define ONLINE_CPLEX_H

#include <iostream>
#include <ilcplex/ilocplex.h>
#include <algorithm>
#include <filesystem>
#include <experimental/filesystem>
#include <stdexcept>
#include <chrono>
#include <string>
#include <vector>
#include <utility>
#include <random>

using namespace std;
namespace fs = std::experimental::filesystem;


typedef struct {
    float* soc_jf;
    float* e_t;
    bool** x_jt;
    bool* y_j;
    bool* u_j;
    bool solved;
} Result;

extern "C" {
Result* solve(float alpha, float beta, float gamma, int current_ts, int T, int m, int nb_past, int nb_assigned, int nb_requests, float w, float eta, float w_G, float tau, float* b_ptr, int* d_j_ptr, int* r_j_ptr, float* soc_0_ptr, float* pv_ptr, float* pr_ptr,
              bool** past_x_jt_ptr_ptr, bool** past_z_jk_ptr_ptr, float* past_soc_jf_ptr, bool* past_y_j_ptr, float* past_e_t_ptr, bool* past_u_j_ptr);
void destroy_result(Result* res, int n);
}


#endif // ONLINE_CPLEX_H
