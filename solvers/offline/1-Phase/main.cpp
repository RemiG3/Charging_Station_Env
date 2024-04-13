#include <iostream>
#include <ilcplex/ilocplex.h>
#include <algorithm>
#include <filesystem>
#include <experimental/filesystem>
//#include <direct.h>
#include <stdexcept>
#include <chrono>
#include <string>
#include <vector>
#include <utility>
#include <random>

using namespace std;
namespace fs = std::experimental::filesystem;

ILOSTLBEGIN

string get_current_working_dir() {
    std::string buff = fs::current_path().string();
    string current_working_dir(buff);
    return current_working_dir;
}


void read_solar_production_signal(int scenario, int n, vector<float> &pv){
    string dir_path = get_current_working_dir();
    dir_path = ".";
    string scenario_file_path;
    if(n <= 0)
        scenario_file_path =  dir_path + "/PV-"+ to_string(scenario)+".csv";
    else
        scenario_file_path =  dir_path + "/PV-"+ to_string(n) + "-" + to_string(scenario) + ".csv";
    cout << "Opening file " << scenario_file_path <<  endl;
    fstream fout;
    string line;
    ifstream myFile(scenario_file_path);
    try {
        if (!myFile.is_open())
            throw  runtime_error("Could not open file " + scenario_file_path);
    } catch (const runtime_error &error) {
        cout << error.what()<<  endl;
    }

    string row;
    int idxRow =0;
    while ( getline(myFile, row)) {
        stringstream check1(row);
        string intermediate;
        int idxColumn = 0;

        while ( getline(check1, intermediate, ',')) {
            if (idxRow != 0) {
                switch(idxColumn) {
                    case 1:
                        pv.push_back( stof(intermediate) );
                        break;
                    default:
                        break;
                }

            }
            idxColumn++;
        }
        idxRow++;
    }
}

void read_price_signal(int scenario, int n, vector<float> &pr){
    string dir_path = get_current_working_dir();
    dir_path  = ".";
    string scenario_file_path;
    if(n <= 0)
        scenario_file_path =  dir_path + "/PR-"+ to_string(scenario)+".csv";
    else
        scenario_file_path =  dir_path + "/PR-"+ to_string(n) + "-" + to_string(scenario) + ".csv";
    cout << "Opening file " << scenario_file_path <<  endl;
    fstream fout;
    string line;
    ifstream myFile(scenario_file_path);
    try {
        if (!myFile.is_open())
            throw  runtime_error("Could not open file " + scenario_file_path);
    } catch (const runtime_error &error) {
        cout << error.what()<<  endl;
    }

    string row;
    int idxRow =0;
    while ( getline(myFile, row)) {
        stringstream check1(row);
        string intermediate;
        int idxColumn = 0;

        while ( getline(check1, intermediate, ',')) {
            if (idxRow != 0) {
                switch(idxColumn) {
                    case 1:
                        pr.push_back( stof(intermediate) );
                        break;
                    default:
                        break;
                }

            }
            idxColumn++;
        }
        idxRow++;
    }
}

void read_ev_scenario(int scenario, int &n, vector<int> &r_j, vector<int> &d_j, vector<float>  &soc_0, vector<float> &b, vector<int> &charger){
    string dir_path = get_current_working_dir();
    dir_path  = ".";
    string scenario_file_path;
    if (n <= 0)
        scenario_file_path =  dir_path + "/ev_scenario-"+ to_string(scenario)+".csv";
    else
        scenario_file_path =  dir_path + "/ev_scenario-"+to_string(n)+"-"+ to_string(scenario)+".csv";
    cout << "Opening file " << scenario_file_path <<  endl;
    fstream fout;
    string line;
    ifstream myFile(scenario_file_path);
    try {
        if (!myFile.is_open())
            throw  runtime_error("Could not open file " + scenario_file_path);
    } catch (const runtime_error &error) {
        cout << error.what()<<  endl;
    }

    string row;
    int idxRow =0;
    while ( getline(myFile, row)) {
        stringstream check1(row);
        string intermediate;
        int idxColumn = 0;

        while ( getline(check1, intermediate, ',')) {
            if (idxRow != 0) {
                switch(idxColumn) {
                    case 1:
                        r_j.push_back( stoi(intermediate) );
                        break;
                    case 2:
                        d_j.push_back( stoi(intermediate) );
                        break;
                    case 3:
                        soc_0.push_back( stof(intermediate) );
                        break;
                    case 4:
                        b.push_back( stoi(intermediate) );
                        break;
                    case 5:
                        charger.push_back( stoi(intermediate) );
                        break;
                    default:
                        break;
                }

            }
            idxColumn++;
        }
        idxRow++;
    }
    n = --idxRow;
}

void read_station_config(int scenario, int &m, float &w, float &eta, float &w_G, float &tau){
    string dir_path = get_current_working_dir();
    dir_path = ".";
    string scenario_file_path =  dir_path + "/station_"+ to_string(scenario)+".csv";
    cout << "Opening file " << scenario_file_path <<  endl;
    fstream fout;
    string line;
    ifstream myFile(scenario_file_path);
    try {
        if (!myFile.is_open())
            throw  runtime_error("Could not open file " + scenario_file_path);
    } catch (const runtime_error &error) {
        cout << error.what()<<  endl;
    }

    string row;
    int idxRow =0;
    while ( getline(myFile, row)) {
        stringstream check1(row);
        string intermediate;
        int idxColumn = 0;

        while ( getline(check1, intermediate, ',')) {
            if (idxRow != 0) {
                switch(idxColumn) {
                    case 1:
                        m = stoi(intermediate);
                        break;
                    case 2:
                        w = stof(intermediate);
                        break;
                    case 3:
                        eta = stof(intermediate);
                        break;
                    case 4:
                        w_G = stof(intermediate);
                        break;
                    case 5:
                        tau = stof(intermediate);
                        break;
                    default:
                        break;
                }

            }
            idxColumn++;
        }
        idxRow++;
    }
}


IloModel create_model(IloEnv const &env, int T, int m , int n, double w, double eta, double w_G, float tau, vector<int> const&r_j, vector<int> const&d_j, vector<float>  const&soc_0, vector<float> const&b, vector<float> const&pv_t, vector<float> const&pr_t,
                      IloArray<IloBoolVarArray> &x_jt, IloArray<IloBoolVarArray> &z_jk, IloNumVarArray &soc_jf, IloBoolVarArray &y_j, IloNumVarArray &e_t, IloNumVarArray &q_j, IloNumVarArray &floor_qj, IloBoolVarArray &u_j) {
    try {
        IloModel mod ( env );
        stringstream name;

        float epsilon = .00001;
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
                if((t >= r_j[j]) && (t < d_j[j]))
                    x_jt[j][t] = IloBoolVar(env, 0, 1, name.str().c_str());
                else
                    x_jt[j][t] = IloBoolVar(env, 0, 0, name.str().c_str());
                name.str("");
            }
        }

        // Create variables z_jk[0][0], ..., z_jk[n-1][j-1]
        for(int j = 1; j < n; ++j) {
            z_jk[j] = IloBoolVarArray(env, j);
            for(int k = 0; k < j; ++k) {
                name << "z_jk_" << j << "-" << k;
                if((d_j[k] <= d_j[j]) && (r_j[j] < d_j[k]))
                    z_jk[j][k] = IloBoolVar(env, 0, 1, name.str().c_str());
                else
                    z_jk[j][k] = IloBoolVar(env, 0, 0, name.str().c_str());
                name.str("");
            }
        }

        // Create variables soc_jf[0], ..., soc_jf[n-1]
        for(int j = 0; j < n; j++) {
            name << "soc_" << j << "_f";
            soc_jf[j] = IloNumVar(env, soc_0[j], 1., IloNumVar::Float, name.str().c_str());
            name.str("");
        }

        // Create variables e_t[0], ..., e_t[T-1]
        for(int t = 0; t < T; t++) {
            name << "e_t_" << t;
            e_t[t] = IloNumVar(env, 0., w_G*tau, IloNumVar::Float, name.str().c_str());       // Constraint (13)
            name.str("");
        }

        // Create variables y_j[0], ..., y_j[n-1]
        for(int j = 0; j < n; j++) {
            name << "y_j_" << j;
            y_j[j] = IloBoolVar(env, 0, 1, name.str().c_str());
            name.str("");
        }

        // Create variables qj[0], ..., qj[n-1]
        for(int j = 0; j < n; j++) {
            name << "q_j_" << j;
            q_j[j] = IloNumVar(env, 0, j+1, IloNumVar::Int, name.str().c_str());
            name.str("");
        }

        // Create variables floor_qj[0], ..., floor_qj[n-1]
        for(int j = 0; j < n; j++) {
            name << "floor_qj_" << j;
            floor_qj[j] = IloNumVar(env, 0, j+1, IloNumVar::Int, name.str().c_str());
            name.str("");
        }

        // Create variables u_j[0], ..., u_j[n-1]
        for(int j = 0; j < n; j++) {
            name << "u_j_" << j;
            u_j[j] = IloBoolVar(env, 0, 1, name.str().c_str());
            name.str("");
        }


        /////////////////////
        //// Constraints ////
        /////////////////////

        IloExpr expr(env), expr1(env);
        expr.clear(); expr1.clear();

        for(int j = 0; j < n; ++j){
            if(j >= 1) {
                for(int k = 0; k < j; ++k) {
                    if((j != k) && (d_j[k] <= d_j[j]) && (r_j[j] < d_j[k])) {
                        expr.clear();
                        for(int t = r_j[j]; t < d_j[k]; ++t)
                            expr += x_jt[j][t];
                        mod.add( 1 - z_jk[j][k] <= IloAbs((q_j[j] - q_j[k]) - m * (floor_qj[j] - floor_qj[k])) );           // Constraint (4)
                        mod.add( IloAbs((q_j[j] - q_j[k]) - m * (floor_qj[j] - floor_qj[k])) <= m * (1 - z_jk[j][k]) );     // Constraint (5)
                        mod.add( expr <= (2 - z_jk[j][k] - y_j[k]) * T );                                                   // Constraint (6)
                    } else {
                        mod.add( z_jk[j][k] == 0 ); // Set default value
                    }
                }
            }

            mod.add( floor_qj[j] <= q_j[j]/m );                                 // Constraint (2)
            mod.add( floor_qj[j] + 1 >= q_j[j]/m + epsilon);                    // Constraint (3)

            expr.clear(); expr1.clear();
            for(int t = r_j[j]; t < d_j[j]; ++t){
                expr += (1 - x_jt[j][t]);
                expr1 += x_jt[j][t];
            }
            mod.add( expr <= (((d_j[j] - r_j[j])*(d_j[j] - r_j[j])) / (((b[j] * (.8 - soc_0[j])) / (w * eta)) * tau + (d_j[j] - r_j[j]))) * y_j[j]  +  T * (1 - y_j[j]) );
                                                                                // Constraint (7)
            mod.add( expr1 <= T * y_j[j] );                                     // Constraint (8)

            //expr.clear();
            //for(int t = r_j[j]; t < d_j[j]; ++t)
            //    expr += x_jt[j][t];
            //mod.add( y_j[j] <= expr );

            expr.clear();
            for(int k = 0; k <= j; ++k)
                expr += y_j[k];
            mod.add(q_j[j] == expr);                                            // Constraint (1)

            expr.clear();
            for(int t = 0; t < T; t++)
                expr += x_jt[j][t];
            mod.add( soc_jf[j] == (soc_0[j] + (tau * w * eta * expr) / b[j]) ); // Constraint (9)
            mod.add( soc_jf[j] <= 1. - .20000001 * (1 - u_j[j]) );              // Constraint (10)
            mod.add( soc_jf[j] >= .8 * u_j[j] );                                // Constraint (11)
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

        ////////////////////////////
        //// Objective Function ////
        ////////////////////////////

        expr.clear();
        for(int t = 0; t < T; t++)
            expr -= e_t[t] * pr_t[t];
        for(int j = 0; j < n; ++j)
            expr += y_j[j] * 10 + u_j[j] * 100;

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


void write_solution_under_online_formulation(string output_filename, int n, int m, int T, vector<int> r_j, IloEnv env, IloCplex cplex, IloArray<IloBoolVarArray> x_jt, vector<vector<float>> y_ij){
    string dir_path = get_current_working_dir();
    string result_file_path =  dir_path + "/" + output_filename;
    fs::create_directory(dir_path);

    cout << "Writing file " << result_file_path <<  endl;
    ofstream ofs(result_file_path);
    IloNumArray vals(env);
    ofs << m << " " << T << "\n";

    string times;
    for(int i = 0; i < m; i++){
        ofs << i << ":";
        times = "";
        for(int t = 0; t < T; t++){
            bool accepted_arrival_on_i_at_t = false;
            for(int j = 0; (j < n) && (!accepted_arrival_on_i_at_t); j++){
                if((r_j[j] == t) && (y_ij[i][j] == 1.))
                    accepted_arrival_on_i_at_t = true;
            }
            if(accepted_arrival_on_i_at_t)
                times += "1,";
            else
                times += "0,";
        }
        if(times.size() > 1){
            times.pop_back();
            ofs << times << "\n";
        }
    }

    vector<vector<vector<float>>> X = *new vector<vector<vector<float>>>();
    for(int i = 0; i < m; i++){
        X.push_back(*new vector<vector<float>>());
        for(int j = 0; j < n; j++){
            X[i].push_back(*new vector<float>());
            cplex.getValues(vals, x_jt[j]);
            for(int t = 0; t < T; t++)
                X[i][j].push_back( vals[t] * y_ij[i][j] );
        }
    }

    for(int i = 0; i < m; i++){
        ofs << i << ":";
        times = "";
        for(int t = 0; t < T; t++){
            int sum_power = 0;
            for(int j = 0; j < n; j++){
                sum_power += int(round(X[i][j][t]));
            }
            times += to_string(sum_power) + ",";
        }
        if(times.size() > 1){
            times.pop_back();
            ofs << times << "\n";
        }
    }

    ofs.close();
}


void write_log_file(IloEnv env, IloCplex cplex, string output_filename, int n, int m, int T, double w, double eta, double w_G, float tau, vector<vector<float>> yij_assigned, vector<int> const&r_j, vector<int> const&d_j, vector<float>  const&soc_0, vector<float> const&b, vector<int> const&charger, vector<float> const&pv, vector<float> const&pr,
                    IloArray<IloBoolVarArray>  &x_jt, IloNumVarArray &soc_jf, IloNumVarArray &e_t, IloBoolVarArray &y_j,
                    IloNum total_time, bool solved, double obj_val, double gap){
    string dir_path = get_current_working_dir();
    string log_filename = output_filename.substr(0, output_filename.find_last_of("."));
    string result_file_path =  dir_path + "/" + log_filename + ".log";
    fs::create_directory(dir_path);

    cout << "Writing file " << result_file_path <<  endl;
    ofstream ofs(result_file_path);

    ofs << "Parameters: \n";
    ofs << "n=" << n << "\n";
    ofs << "m=" << m << "\n";
    ofs << "T=" << T << "\n";
    ofs << "P=" << w << "\n";
    ofs << "eta=" << eta << "\n";
    ofs << "tau=" << tau << "\n";
    ofs << "w_G=" << w_G << "\n";

    string list_pr = "price=";
    for(int t = 0; t < T; ++t)
        list_pr += to_string(pr[t]) + ",";
    list_pr.pop_back();
    ofs << list_pr << "\n";

    string list_pv = "pv=";
    for(int t = 0; t < T; ++t)
        list_pv += to_string(pv[t]) + ",";
    list_pv.pop_back();
    ofs << list_pv << "\n";

    string list_soc = "SOC_0=";
    for(int j = 0; j < n; ++j)
        list_soc += to_string(soc_0[j]) + ",";
    list_soc.pop_back();
    ofs << list_soc << "\n";

    string list_bmax = "Bmax=";
    for(int j = 0; j < n; ++j)
        list_bmax += to_string(b[j]) + ",";
    list_bmax.pop_back();
    ofs << list_bmax << "\n";

    ofs << "arrivals=";
    string list_arr = "";
    for(int j = 0; j < n; ++j)
        list_arr += to_string(r_j[j]) + ",";
    list_arr.pop_back();
    ofs << list_arr << "\n";

    ofs << "departures=";
    string list_dep = "";
    for(int j = 0; j < n; ++j)
        list_dep += to_string(d_j[j]) + ",";
    list_dep.pop_back();
    ofs << list_dep << "\n";

    ofs << "Decision variables:\n";
    IloNumArray vals(env);
    ofs << "X_ijt:\n";
    for(int i = 0; i < m; i++){
        ofs << i << ":\n";
        for(int j = 0; j < n; j++){
            string list_ev = "";
            ofs << j << ":";
            cplex.getValues(vals, x_jt[j]);
            for(int t = 0; t < T; t++){
                list_ev += to_string(int(round(vals[t]*yij_assigned[i][j]))) + ",";
            }
            list_ev.pop_back();
            ofs << list_ev << "\n";
        }
    }

    ofs << "y_ij:\n";
    for(int i = 0; i < m; i++){
        ofs << i << ":";
        string list_ev = "";
        for(int j = 0; j < n; j++){
            if(yij_assigned[i][j] == 1)
                list_ev += to_string(j) + ",";
        }
        if(list_ev.length() > 0)
            list_ev.pop_back();
        ofs << list_ev << "\n";
    }

    cplex.getValues(vals, soc_jf);
    string list_socf = "SOC_f=";
    for(int j = 0; j < n; j++){
        list_socf += to_string(vals[j]) + ",";
    }
    list_socf.pop_back();
    ofs << list_socf << "\n";

    cplex.getValues(vals, e_t);
    string list_gt = "e_t=";
    for(int t = 0; t < T; t++){
        list_gt += to_string(vals[t]) + ",";
    }
    list_gt.pop_back();
    ofs << list_gt << "\n";


    ofs << "Additional infos: \n";
    ofs << "GAP=" << to_string(gap) << "\n";
    ofs << "computation_time=" << to_string(total_time) << "\n";
    ofs << "obj_val=" << to_string(obj_val) << "\n";
    ofs << "solved=" << to_string(solved) << "\n";

    ofs.close();
}


void run_model(int nb_ev, int ev_scenario, int pr_scenario, int pv_scenario, int station_scenario, bool use_charger_assignement, string output_filename){
    IloEnv env;
    IloModel mod;

    // Parameters of the problem
    float tau;
    int n = nb_ev;
    int m = 0;
    float w = 0;
    float eta = 0;
    float w_G = 0;

    // Read scenario from csv file
    vector<int> r_j;
    vector<int> d_j;
    vector<int> charger;
    vector<float> soc_0;
    vector<float> b;
    vector<float> pv;
    vector<float> pr;

    read_solar_production_signal(pv_scenario, n, pv);
    read_price_signal(pr_scenario, n, pr);
    read_ev_scenario(ev_scenario, n, r_j, d_j, soc_0, b, charger);
    read_station_config(station_scenario, m, w, eta, w_G, tau);

    //////////////////////
    /// Pre-processing ///
    //////////////////////

    // Combine the lists (arrivaT, d_j, soc_0) into a vector of tuples
    std::vector<std::tuple<int, int, float>> combined;
    for (size_t j = 0; j < d_j.size(); ++j) {
        combined.push_back(std::make_tuple(d_j[j], r_j[j], soc_0[j]));
    }
    // Sort the combined list based on the first element of each tuple
    std::sort(combined.begin(), combined.end());
    for (size_t j = 0; j < combined.size(); ++j) {
        d_j[j] = std::get<0>(combined[j]);
        r_j[j] = std::get<1>(combined[j]);
        soc_0[j] = std::get<2>(combined[j]);
    }

    ///////////////////////
    //// Print problem ////
    ///////////////////////

    cout <<"\tEV" << "\tArrival times" << "\tDeparture times" <<"\t\tBattery capacity"  << "\tInitial SOC" << "\t\tAssociated charger" << "\n";
    for(int j=0; j<n; j++){
        cout <<"\t" << j << "\t" << setw(sizeof("Arrival times")-1) <<   right <<  r_j[j]
             << "\t" << setw(sizeof("Departure times")-1) <<   right <<  d_j[j]
             << "\t\t" << setw(sizeof("Battery capacity")-1) <<   right <<  b[j]
             << "\t" << setw(sizeof("Initial SOC")-1) <<   right << soc_0[j]
             << "\t\t" << setw(sizeof("Desired SOC")-1) <<   right << charger[j] <<"\n";
    }

    cout << "\nNumber of chargers: " << m << "\nMaximum power rate delivered by the chargers: " << w
         << "\nEfficiency of the charging: " << eta <<"\nGrid limit: " << w_G << "\nTau (step time): " << tau << "\n";

    cout << "\nElectricity price signal: " << endl;
    for (float e : pr){
        cout << e << ", ";
    }
    cout << "\n";

    cout << "\nSolar production signal: " << endl;
    for (float e : pv){
        cout << e << ", ";
    }
    cout << "\n" << endl;

    int T = int(round(24*1/tau)); // Number of timesteps
    vector<vector<float>> yij_assigned;
    IloNumVarArray soc_jf(env, n);
    IloNumVarArray e_t(env, T);
    IloArray<IloBoolVarArray> x_jt (env, n);
    IloArray<IloBoolVarArray> z_jk (env, n);
    IloBoolVarArray y_j (env, n);
    IloNumVarArray q_j (env, n);
    IloNumVarArray floor_qj (env, n);
    IloBoolVarArray u_j (env, n);

    IloNum elapsed_time = 0.;
    bool solved = false, solved_log = false;
    if(!use_charger_assignement)
        charger.clear();
    int timeout = 30*60;

    mod = create_model(env, T, m, n, w, eta, w_G, tau, r_j, d_j, soc_0, b, pv, pr, x_jt, z_jk, soc_jf, y_j, e_t, q_j, floor_qj, u_j);
    IloCplex cplex(mod);

    try {
        // Try to solve with CPLEX (and hope it does not raise an exception!)
        cplex.setParam(IloCplex::Param::TimeLimit, timeout);
        IloNum start_time = cplex.getCplexTime();
        solved = cplex.solve();
        elapsed_time = (cplex.getCplexTime()-start_time);
    } catch(const IloException& e) {
        cerr << "\n\nCPLEX Raised an exception:\n";
        cerr << e << "\n";
        env.end();
        throw;
    }

    if(solved) {
        // If CPLEX successfully solved the model, print the results
        cout << "\tStatus: " << cplex.getStatus() << "\n";
        cout << "\tObjective value: " << cplex.getObjValue() << "\n";
        cout << "\tSolution found!" << endl;


        ////////////////////////
        //// Print solution ////
        ////////////////////////

        IloNumArray vals(env);
        cplex.getValues(vals, soc_jf);
        cout << " soc_jf = " << vals << endl;
        cplex.getValues(vals, y_j);
        cout << " y_j = " << vals << endl;

        double total_price = 0.;
        cplex.getValues(vals, e_t);
        cout << " e_t = " << vals << endl;
        for(int t = 0; t < T; t++)
            total_price += vals[t] * pr[t];
        cout << " Total Price = " << total_price << endl;
        //cout << tmp_price << endl;

        ////////////////////
        /// Post-process ///
        ////////////////////

        for(int i = 0; i < m; i++){
            yij_assigned.push_back(*new vector<float>());
            for(int j = 0; j < n; ++j){
                yij_assigned[i].push_back(0);
            }
        }

        // EV-charger association
        cplex.getValues(vals, y_j);
        int i = 0;
        for(int j = 0; j < n; ++j){
            if(int(round(vals[j])) != 0){
                charger[j] = i;
                yij_assigned[i][j] = 1;
                i = (i+1) % m;
            } else {
                yij_assigned[i][j] = 0;
            }
        }

        solved_log = solved && (elapsed_time < timeout);

        write_solution_under_online_formulation(output_filename, n, m, T, r_j, env, cplex, x_jt, yij_assigned);
        write_log_file(env, cplex, output_filename, n, m, T, w, eta, w_G, tau, yij_assigned, r_j, d_j, soc_0, b, charger, pv, pr,
                            x_jt, soc_jf, e_t, y_j,
                            elapsed_time, solved_log, (double)cplex.getBestObjValue(), (double)cplex.getMIPRelativeGap());

    } else {
        cerr << "\n\nCplex error!\n";
        cerr << "\tStatus: " << cplex.getStatus() << "\n";
        cerr << "\tSolver status: " << cplex.getCplexStatus() << "\n";
    }

    x_jt.end();
    z_jk.end();
    y_j.end();
    q_j.end();
    floor_qj.end();
    soc_jf.end();
    e_t.end();
    env.end();
}


int main(int argc, char* argv[]) {
    int scenario = 0, nb_ev = -1, ev_scenario = -1, pr_scenario = -1, pv_scenario = -1, station_scenario = -1;
    bool use_charger_assignement = false;
    string output_filename = "";

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--scenario") {
            if (i+1 < argc) {
                scenario = atoi(argv[++i]);
            }
        }
        if (std::string(argv[i]) == "--ev_scenario") {
            if (i+1 < argc) {
                ev_scenario = atoi(argv[++i]);
            }
        }
        if (std::string(argv[i]) == "--nb_ev") {
            if (i+1 < argc) {
                nb_ev = atoi(argv[++i]);
            }
        }
        if (std::string(argv[i]) == "--pr_scenario") {
            if (i+1 < argc) {
                pr_scenario = atoi(argv[++i]);
            }
        }
        if (std::string(argv[i]) == "--pv_scenario") {
            if (i+1 < argc) {
                pv_scenario = atoi(argv[++i]);
            }
        }
        if (std::string(argv[i]) == "--station_scenario") {
            if (i+1 < argc) {
                station_scenario = atoi(argv[++i]);
            }
        }
        if (std::string(argv[i]) == "--output_filename") {
            output_filename = argv[++i];
        }
    }

    if(output_filename == "")
        output_filename = "online_solution_formulation_" + to_string(scenario) + ".txt";

    if(scenario >= 0){
        ev_scenario = (ev_scenario >= 0) ? ev_scenario : scenario;
        pr_scenario = (pr_scenario >= 0) ? pr_scenario : scenario;
        pv_scenario = (pv_scenario >= 0) ? pv_scenario : scenario;
        station_scenario = (station_scenario >= 0) ? station_scenario : scenario;
    }

    run_model(nb_ev, ev_scenario, pr_scenario, pv_scenario, station_scenario, use_charger_assignement, output_filename);

    return 0;
}
