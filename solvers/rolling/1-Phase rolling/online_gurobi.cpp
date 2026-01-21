#include "online_gurobi.h"



Result* solve(float alpha, float beta, float gamma, int current_ts, int T, int m, int nb_past, int nb_assigned, int nb_requests, float w, float eta, float w_G, float tau, float* b_ptr, int* d_j_ptr, int* r_j_ptr, float* soc_0_ptr, float* pv_ptr, float* pr_ptr,
              bool*** past_x_ijt_ptr_ptr, float* past_soc_jf_ptr, bool** past_y_ij_ptr, float* past_e_t_ptr, bool* past_u_j_ptr, bool* assigned_ev_ptr, bool* past_ev_ptr) {
    GRBEnv env = GRBEnv(true);
    env.start();

    GRBModel model = GRBModel(env);
    stringstream name;

    ////////////////////////////
    //////// Parameters ////////
    ////////////////////////////

    vector<float> b;
    vector<int> d_j;
    vector<int> r_j;
    vector<float> soc_0;
    vector<float> pv_t;
    vector<float> pr_t;
    vector<vector<vector<bool>>> past_x_ijt;
    vector<float> past_soc_jf;
    vector<vector<bool>> past_y_ij;
    vector<float> past_e_t;
    vector<bool> past_u_j;
    vector<bool> assigned_ev;
    vector<bool> past_ev;

    int n = nb_requests;
    for(int j = 0; j < n; j++){
        r_j.push_back(*(r_j_ptr+j));
        d_j.push_back(*(d_j_ptr+j));
        b.push_back(*(b_ptr+j));
        soc_0.push_back(*(soc_0_ptr+j));
        past_soc_jf.push_back(*(past_soc_jf_ptr+j));
        past_u_j.push_back(*(past_u_j_ptr+j));
        assigned_ev.push_back(*(assigned_ev_ptr+j));
        past_ev.push_back(*(past_ev_ptr+j));
    }

    for(int i = 0; i < m; i++){
        vector<vector<bool>> inter_x_ijt;
        vector<bool> inter_y_ij;
        for(int j = 0; j < n; j++){
            vector<bool> inter_x_ij;
            for(int t = 0; t < T; t++)
                inter_x_ij.push_back(*(*(*(past_x_ijt_ptr_ptr+i)+j)+t));
            inter_x_ijt.push_back(inter_x_ij);
            inter_y_ij.push_back(*(*(past_y_ij_ptr+i)+j));
        }
        past_y_ij.push_back(inter_y_ij);
        past_x_ijt.push_back(inter_x_ijt);
    }

    for(int t = 0; t < current_ts; t++)
        past_e_t.push_back(*(past_e_t_ptr+t));

    for(int t = 0; t < T; t++){
        pv_t.push_back(*(pv_ptr+t));
        pr_t.push_back(*(pr_ptr+t));
    }

    float epsilon = 1e-4;
    float obj_coef = 0.;
    vector<vector<bool>> o_jt;
    for(int j = 0; j < n; ++j) {
        vector<bool> temp_vector;
        for(int t = 0; t < T; ++t){
            temp_vector.push_back((t >= r_j[j]) && (t < d_j[j]));
        }
        o_jt.push_back(temp_vector);
    }


    ////////////////////////////
    //// Decision Variables ////
    ////////////////////////////

    vector<GRBVar> soc_jf(n);
    vector<GRBVar> e_t(T);
    vector<vector<vector<GRBVar>>> x_ijt(m, vector<vector<GRBVar>>(n, vector<GRBVar>(T)));
    vector<vector<GRBVar>> y_ij(m, vector<GRBVar>(n));
    vector<GRBVar> u_j(n);

    // Create variables x_ijt[0][0][0], ..., x_ijt[m-1][n-1][T-1]
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; ++j) {
            for(int t = 0; t < T; t++) {
                name << "x_ijt_" << i << "-" << j << "-" << t;
                if(t < current_ts) {
                    x_ijt[i][j][t] = model.addVar(past_x_ijt[i][j][t], past_x_ijt[i][j][t], obj_coef, GRB_BINARY, name.str());
                } else {
                    if((t >= r_j[j]) && (t < d_j[j]))
                        x_ijt[i][j][t] = model.addVar(0., 1., obj_coef, GRB_BINARY, name.str());
                    else
                        x_ijt[i][j][t] = model.addVar(0., 0., obj_coef, GRB_BINARY, name.str());
                }
                name.str("");
            }
        }
    }

    // Create variables soc_jf[0], ..., soc_jf[n-1]
    for(int j = 0; j < n; j++) {
        name << "soc_" << j << "_f";
        //if(j < nb_past)
        if(past_ev[j])
            soc_jf[j] = model.addVar(past_soc_jf[j]-epsilon, past_soc_jf[j]+epsilon, obj_coef, GRB_CONTINUOUS, name.str());
        else
            soc_jf[j] = model.addVar(soc_0[j], 1., obj_coef, GRB_CONTINUOUS, name.str());
        name.str("");
    }

    // Create variables e_t[0], ..., e_t[T-1]
    for(int t = 0; t < T; t++) {
        name << "e_t_" << t;
        if(t < current_ts)
            e_t[t] = model.addVar(max(0.f, past_e_t[t]-epsilon), min(past_e_t[t]+epsilon, w_G*tau-epsilon), obj_coef, GRB_CONTINUOUS, name.str());
        else
            e_t[t] = model.addVar(0., w_G*tau-epsilon, obj_coef, GRB_CONTINUOUS, name.str());       // Constraint (9)
        name.str("");
    }

    // Create variables y_ij[0][0], ..., y_ij[m-1][n-1]
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            name << "y_ij_" << i << "-" << j;
            //if(j < nb_past+nb_assigned) {
            if(assigned_ev[j]) {
                y_ij[i][j] = model.addVar(past_y_ij[i][j], past_y_ij[i][j], obj_coef, GRB_BINARY, name.str());
            } else
                y_ij[i][j] = model.addVar(0., 1., obj_coef, GRB_BINARY, name.str());
            name.str("");
        }
    }

    // Create variables u_j[0], ..., u_j[n-1]
    for(int j = 0; j < n; j++) {
        name << "u_j_" << j;
        if(past_ev[j])
            u_j[j] = model.addVar(past_u_j[j], past_u_j[j], obj_coef, GRB_BINARY, name.str());
        else
            u_j[j] = model.addVar(0., 1., obj_coef, GRB_BINARY, name.str());
        name.str("");
    }


    /////////////////////
    //// Constraints ////
    /////////////////////

    GRBLinExpr expr, expr1;
    for(int j = 0; j < n; ++j){
        for(int i = 0; i < m; i++){
            // Constraint (1)
            expr.clear();
            for(int j_ = 0; j_ < n; j_++) {
                if((j_ != j) && (r_j[j_] < d_j[j]) && (d_j[j] <= d_j[j_])) {
                    for(int t = r_j[j_]; t < d_j[j]; t++)
                        expr += x_ijt[i][j_][t];
                }
            }
            model.addConstr( y_ij[i][j] + 1./(n * T) * expr <= 1. );

            for(int j_ = 0; j_ < n; ++j_){
                if((j_ != j) && (((r_j[j_] < r_j[j]) && (d_j[j_] > d_j[j])) || ((r_j[j_] <= r_j[j]) && (d_j[j_] == d_j[j])))){
                    model.addConstr( y_ij[i][j] + y_ij[i][j_] <= 1 );          // Added for online rolling only
                }
            }
        }

        for(int i = 0; i < m; i++) {
            expr1.clear();
            for(int t = 0; t < T; ++t){
                expr1 += x_ijt[i][j][t];
            }
            model.addConstr( expr1 <= T * y_ij[i][j] );                       // Constraint (3)
        }

        for(int t = 0; t < T; ++t){
            expr.clear();
            for(int i = 0; i < m; ++i){
                expr += x_ijt[i][j][t];
            }
            model.addConstr( expr <= 1 );                                    // Tigheting Constraint (4)
        }

        expr.clear();
        for(int i = 0; i < m; i++)
            expr += y_ij[i][j];
        model.addConstr( expr <= 1 );                                       // Constraint (4)

        expr.clear();
        for(int i = 0; i < m; i++) {
            for(int t = 0; t < T; t++)
                expr += x_ijt[i][j][t];
        }
        model.addConstr( soc_jf[j] == (soc_0[j] + (tau * w * eta * expr) / b[j]) ); // Constraint (5)
        model.addConstr( soc_jf[j] <= 1. - .20000001 * (1 - u_j[j]) );              // Constraint (6)
        model.addConstr( soc_jf[j] >= .8 * u_j[j] );                                // Constraint (7)
    }

    for(int t = 0; t < T; t++){
        expr.clear(); expr1.clear();
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                expr += y_ij[i][j] * o_jt[j][t];
                expr1 += x_ijt[i][j][t];
            }
        }
        model.addConstr( e_t[t] >= tau * (expr1 * w - pv_t[t]) );                   // Constraint (8)
        model.addConstr( expr <= 2 * m );                                           // Constraint (10)
    }


    ///////////////////////////
    /// Objective Function  ///
    ///////////////////////////

    expr.clear();
    for(int t = 0; t < T; t++)
        expr -= (e_t[t] * pr_t[t]) * alpha;

    for(int j = 0; j < n; ++j) {
        expr1.clear();
        for(int i = 0; i < m; i++)
            expr1 += y_ij[i][j];
        expr += expr1 * beta + u_j[j] * gamma;
    }

    model.setObjective(expr, GRB_MAXIMIZE);


    /////////////////
    //// Solving ////
    /////////////////

    int timeout = 5*60; // 5 minutes for rolling solver
    bool solved = false;

    try {
        model.set(GRB_IntParam_OutputFlag, 0);
        model.set(GRB_DoubleParam_TimeLimit, timeout);
        double start_time = model.get(GRB_DoubleAttr_Runtime);
        model.optimize();
        double elapsed_time = model.get(GRB_DoubleAttr_Runtime) - start_time;
        int status = model.get(GRB_IntAttr_Status);
        solved = (status != GRB_INFEASIBLE);

        if (status == GRB_OPTIMAL) {
            //cout << "Optimal solution found." << endl;
        } else if (status == GRB_TIME_LIMIT && model.get(GRB_IntAttr_SolCount) > 0) {
            //cout << "Solution found within the time limit." << endl;
        } else {
            solved = false;
            cerr << "No feasible solution found." << endl;
        }
    } catch(const GRBException& e) {
        cerr << "\n\nGUROBI Raised an exception:\n";
        cerr << e.getMessage() << "\n";
        throw;
    }


    //////////////////////
    //// Get Solution ////
    //////////////////////

    Result* res = (Result*)malloc(sizeof(Result));
    res->soc_jf = (float *)malloc(n * sizeof(float));
    res->e_t = (float *)malloc(T * sizeof(float));
    res->x_ijt = (bool***)malloc(m * sizeof(bool**));
    res->y_ij = (bool**)malloc(m * sizeof(bool*));
    for(int i = 0; i < m; i++) {
        res->y_ij[i] = (bool*)malloc(n * sizeof(bool));
        res->x_ijt[i] = (bool**)malloc(n * sizeof(bool*));
        for(int j = 0; j < n; ++j)
            res->x_ijt[i][j] = (bool*)malloc(T * sizeof(bool));
    }
    res->u_j = (bool*)malloc(n * sizeof(bool));
    res->solved = solved;

    if(solved) {
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; ++j) {
                for(int t = 0; t < T; ++t){
                    res->x_ijt[i][j][t] = (int(round(x_ijt[i][j][t].get(GRB_DoubleAttr_X))) > 0);
                }
            }
        }
        for(int t = 0; t < T; ++t)
            res->e_t[t] = e_t[t].get(GRB_DoubleAttr_X);

        for(int j = 0; j < n; ++j)
            res->soc_jf[j] = soc_jf[j].get(GRB_DoubleAttr_X);

        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; ++j) {
                res->y_ij[i][j] = (int(round(y_ij[i][j].get(GRB_DoubleAttr_X))) > 0);
            }
        }

        for(int j = 0; j < n; ++j)
            res->u_j[j] = (int(round(u_j[j].get(GRB_DoubleAttr_X))) > 0);
    } else {
        cerr << "\n\nGurobi error!\n";
        cerr << "\tStatus: " << model.get(GRB_IntAttr_Status) << "\n";
    }

    x_ijt.end();
    y_ij.end();
    soc_jf.end();
    e_t.end();
    u_j.end();

    return res;
}


void destroy_result(Result* res, int m, int n, int T) {
    if (res == nullptr) {
        return; // If res is already nullptr, nothing to free
    }
    if (res->x_ijt != nullptr) {
        for (int i = 0; i < m; i++) {
            if (res->x_ijt[i] != nullptr) {
                for (int j = 0; j < n; j++) {
                    if (res->x_ijt[i][j] != nullptr) {
                        free(res->x_ijt[i][j]);
                        res->x_ijt[i][j] = nullptr;
                    }
                }
                free(res->x_ijt[i]);
                res->x_ijt[i] = nullptr;
            }
        }
        free(res->x_ijt);
        res->x_ijt = nullptr;
    }
    if (res->y_ij != nullptr) {
        for (int i = 0; i < m; i++) {
            if (res->y_ij[i] != nullptr) {
                free(res->y_ij[i]);
                res->y_ij[i] = nullptr;
            }
        }
        free(res->y_ij);
        res->y_ij = nullptr;
    }
    if (res->soc_jf != nullptr) {
        free(res->soc_jf);
        res->soc_jf = nullptr;
    }
    if (res->e_t != nullptr) {
        free(res->e_t);
        res->e_t = nullptr;
    }
    if (res->u_j != nullptr) {
        free(res->u_j);
        res->u_j = nullptr;
    }
    free(res);
    res = nullptr;
}

