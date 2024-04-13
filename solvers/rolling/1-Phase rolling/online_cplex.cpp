#include "online_cplex.h"

ILOSTLBEGIN

IloModel create_model(IloEnv const &env, int alpha, int beta, int gamma, int T, int m , int nb_past, int nb_assigned, int nb_requests, float w, float eta, float w_G, float tau, vector<int> const&r_j, vector<int> const&d_j, vector<float>  const&soc_0, vector<float> const&b, vector<float> const&pv_t, vector<float> const&pr_t,
                      int current_ts, vector<vector<bool>> const&past_x_jt, vector<vector<bool>> const&past_z_jk, vector<float> const&past_soc_jf, vector<bool> const&past_y_j, vector<float> const&past_e_t, vector<bool> const&past_u_j,
                      IloArray<IloBoolVarArray> &x_jt, IloNumVarArray &soc_jf, IloBoolVarArray &y_j, IloNumVarArray &e_t, IloBoolVarArray &u_j) {
    try {
        IloModel mod ( env );
        stringstream name;


        ////////////////////////////
        //////// Parameters ////////
        ////////////////////////////

        float epsilon = 1e-5;
        int n = nb_past + nb_assigned + nb_requests;

        vector<vector<bool>> o_jt = *new vector<vector<bool>>();
        for(int j = 0; j < n; ++j) {
            o_jt.push_back( *new vector<bool>() );
            for(int t = 0; t < T; ++t){
                o_jt[j].push_back( (t >= r_j[j]) && (t < d_j[j]) );
            }
        }


        ////////////////////////////
        //// Decision Variables ////
        ////////////////////////////

        // Create variables x_jt[0][0], ..., x_jt[n-1][T-1]
        for(int j = 0; j < n; ++j) {
            x_jt[j] = IloBoolVarArray(env, T);
            for(int t = 0; t < T; t++) {
                name << "x_jt_" << j << "-" << t;
                if(t < current_ts) {
                    x_jt[j][t] = IloBoolVar(env, past_x_jt[j][t], past_x_jt[j][t], name.str().c_str());
                } else {
                    if((t >= r_j[j]) && (t < d_j[j]))
                        x_jt[j][t] = IloBoolVar(env, 0, 1, name.str().c_str());
                    else
                        x_jt[j][t] = IloBoolVar(env, 0, 0, name.str().c_str());
                }
                name.str("");
            }
        }

        // Create variables soc_jf[0], ..., soc_jf[n-1]
        for(int j = 0; j < n; j++) {
            name << "soc_" << j << "_f";
            if(j < nb_past)
                soc_jf[j] = IloNumVar(env, past_soc_jf[j]-epsilon, past_soc_jf[j]+epsilon, IloNumVar::Float, name.str().c_str());
            else
                soc_jf[j] = IloNumVar(env, soc_0[j], 1., IloNumVar::Float, name.str().c_str());
            name.str("");
        }

        // Create variables e_t[0], ..., e_t[T-1]
        for(int t = 0; t < T; t++) {
            name << "e_t_" << t;
            if(t < current_ts)
                e_t[t] = IloNumVar(env, max(0.f, past_e_t[t]-epsilon), min(past_e_t[t]+epsilon, w_G*tau), IloNumVar::Float, name.str().c_str());
            else
                e_t[t] = IloNumVar(env, 0., w_G*tau, IloNumVar::Float, name.str().c_str());       // Constraint (13)
            name.str("");
        }

        // Create variables y_j[0], ..., y_j[n-1]
        for(int j = 0; j < n; j++) {
            name << "y_j_" << j;
            if(j < nb_past+nb_assigned)
                y_j[j] = IloBoolVar(env, past_y_j[j], past_y_j[j], name.str().c_str());
            else
                y_j[j] = IloBoolVar(env, 0, 1, name.str().c_str());
            name.str("");
        }

        // Create variables u_j[0], ..., u_j[n-1]
        for(int j = 0; j < n; j++) {
            name << "u_j_" << j;
            if(j < nb_past)
                u_j[j] = IloBoolVar(env, past_u_j[j], past_u_j[j], name.str().c_str());
            else
                u_j[j] = IloBoolVar(env, 0, 1, name.str().c_str());
            name.str("");
        }


        /////////////////////
        //// Constraints ////
        /////////////////////

        IloExpr expr(env), expr1(env);
        expr.clear();
        expr1.clear();
        for(int j = 0; j < n; ++j){
            for(int k = 0; k < j; ++k) {
                if((j != k) && (d_j[k] <= d_j[j]) && (r_j[j] < d_j[k])) {
                    expr.clear();
                    for(int t = r_j[j]; t < d_j[k]; ++t)
                        expr += x_jt[j][t];
                    mod.add( expr <= (2 - past_z_jk[j][k] - y_j[k]) * T );          // Constraint (6)
                }
            }

            expr.clear();
            for(int t = r_j[j]; t < d_j[j]; ++t)
                expr += x_jt[j][t];
            mod.add( expr <= T * y_j[j] );                                         // Constraint (8)
            mod.add( y_j[j] <= expr );                                             // Replace constraint (7)

            expr.clear();
            for(int t = 0; t < T; t++)
                expr += x_jt[j][t];
            mod.add( soc_jf[j] == (soc_0[j] + (tau * w * eta * expr) / b[j]) );     // Constraint (9)
            mod.add( soc_jf[j] <= 1. - .20000001 * (1 - u_j[j]) );                  // Constraint (10)
            mod.add( soc_jf[j] >= .8 * u_j[j] );                                    // Constraint (11)
        }

        for(int t = 0; t < T; t++){
            expr.clear(); expr1.clear();
            for(int j = 0; j < n; j++) {
                expr += y_j[j] * o_jt[j][t];
                expr1 += x_jt[j][t];
            }
            mod.add( e_t[t] >= tau * (expr1 * w - pv_t[t]) );                   // Constraint (12)
            mod.add( expr <= 2 * m );                                           // Constraint (14)
        }


        ///////////////////////////
        /// Objective Function  ///
        ///////////////////////////

        expr.clear();
        for(int t = 0; t < T; t++)
            expr -= (e_t[t] * pr_t[t]) * alpha;

        for(int j = 0; j < n; ++j)
            expr += y_j[j] * beta + u_j[j] * gamma;


        IloObjective obj(env, expr, IloObjective::Maximize);
        mod.add(obj);

        expr.end();
        expr1.end();

        return mod;

    } catch(const IloException& e) {
        cerr << "\n\nCPLEX Raised an exception:\n";
        cerr << e << "\n";
        throw;
    }
}


Result* solve(float alpha, float beta, float gamma, int current_ts, int T, int m, int nb_past, int nb_assigned, int nb_requests, float w, float eta, float w_G, float tau, float* b_ptr, int* d_j_ptr, int* r_j_ptr, float* soc_0_ptr, float* pv_ptr, float* pr_ptr,
              bool** past_x_jt_ptr_ptr, bool** past_z_jk_ptr_ptr, float* past_soc_jf_ptr, bool* past_y_j_ptr, float* past_e_t_ptr, bool* past_u_j_ptr) {
    IloEnv env;
    IloModel mod;


    ///////////////////////
    //// Print problem ////
    ///////////////////////

    vector<float> b;
    vector<int> d_j;
    vector<int> r_j;
    vector<float> soc_0;
    vector<float> pv_t;
    vector<float> pr_t;
    vector<vector<bool>> past_x_jt;
    vector<vector<bool>> past_z_jk;
    vector<float> past_soc_jf;
    vector<bool> past_y_j;
    vector<float> past_e_t;
    vector<bool> past_u_j;

    int n = nb_past + nb_assigned + nb_requests;
    for(int j = 0; j < n; j++){
        r_j.push_back(*(r_j_ptr+j));
        d_j.push_back(*(d_j_ptr+j));
        b.push_back(*(b_ptr+j));
        soc_0.push_back(*(soc_0_ptr+j));

        vector<bool> inter_z_jk;
        for(int k = 0; k < n; k++)
            inter_z_jk.push_back(*(*(past_z_jk_ptr_ptr+j)+k));
        past_z_jk.push_back(inter_z_jk);

        vector<bool> inter_x_jt;
        for(int t = 0; t < T; t++)
            inter_x_jt.push_back(*(*(past_x_jt_ptr_ptr+j)+t));
        past_x_jt.push_back(inter_x_jt);
        past_y_j.push_back(*(past_y_j_ptr+j));
        past_soc_jf.push_back(*(past_soc_jf_ptr+j));
        past_u_j.push_back(*(past_u_j_ptr+j));
    }

    for(int t = 0; t < current_ts; t++)
        past_e_t.push_back(*(past_e_t_ptr+t));

    for(int t = 0; t < T; t++){
        pv_t.push_back(*(pv_ptr+t));
        pr_t.push_back(*(pr_ptr+t));
    }


    //////////////////////
    ////// Solving ///////
    //////////////////////

    IloNumVarArray soc_jf(env, (n));
    IloNumVarArray e_t(env, T);
    IloArray<IloBoolVarArray> x_jt (env, (n));
    IloBoolVarArray y_j (env, (n));
    IloBoolVarArray u_j (env, (n));

    bool solved = false;
    int timeout = 30*60;
    mod = create_model(env, alpha, beta, gamma, T, m, nb_past, nb_assigned, nb_requests, w, eta, w_G, tau, r_j, d_j, soc_0, b, pv_t, pr_t,
                       current_ts, past_x_jt, past_z_jk, past_soc_jf, past_y_j, past_e_t, past_u_j,
                       x_jt, soc_jf, y_j, e_t, u_j);
    IloCplex cplex(mod);

    try {
        cplex.setParam(IloCplex::Param::TimeLimit, timeout);
        cplex.setOut(env.getNullStream());
        solved = cplex.solve();
    } catch(const IloException& e) {
        cerr << "\n\nCPLEX Raised an exception:\n";
        cerr << e << "\n";
        env.end();
        throw;
    }

    Result* res = (Result*)malloc(sizeof(Result));
    res->soc_jf = (float *)malloc((n) * sizeof(float));
    res->e_t = (float *)malloc(T * sizeof(float));
    res->x_jt = (bool**)malloc((n) * sizeof(bool*));
    for(int j = 0; j < (n); ++j)
        res->x_jt[j] = (bool*)malloc(T * sizeof(bool));
    res->y_j = (bool*)malloc((n) * sizeof(bool));
    res->u_j = (bool*)malloc((n) * sizeof(bool));
    res->solved = (bool)malloc(sizeof(bool));

    res->solved = solved;

    IloNumArray vals(env);
    if(solved) {
        for(int j = 0; j < n; ++j) {
            cplex.getValues(vals, x_jt[j]);
            for(int t = 0; t < T; ++t){
                res->x_jt[j][t] = (int(round(vals[t])) > 0);
            }
        }

        cplex.getValues(vals, e_t);
        for(int t = 0; t < T; ++t)
            res->e_t[t] = vals[t];

        cplex.getValues(vals, soc_jf);
        for(int j = 0; j < n; ++j)
            res->soc_jf[j] = vals[j];

        cplex.getValues(vals, y_j);
        for(int j = 0; j < n; ++j)
            res->y_j[j] = (int(round(vals[j])) > 0);

        cplex.getValues(vals, u_j);
        for(int j = 0; j < n; ++j)
            res->u_j[j] = (int(round(vals[j])) > 0);
    } else {
        cerr << "\n\nCplex error!\n";
        cerr << "\tStatus: " << cplex.getStatus() << "\n";
        cerr << "\tSolver status: " << cplex.getCplexStatus() << "\n";
    }

    x_jt.end();
    y_j.end();
    soc_jf.end();
    e_t.end();
    env.end();

    return res;
}


void destroy_result(Result* res, int n){
    free(res->soc_jf);
    free(res->e_t);
    for(int j = 0; j < n; ++j)
        free(res->x_jt[j]);
    free(res->x_jt);
    free(res->y_j);
    free(res->u_j);
}
