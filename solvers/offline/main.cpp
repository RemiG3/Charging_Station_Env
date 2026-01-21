#include <iostream>
#include <gurobi_c++.h>
#include <algorithm>
#include <filesystem>
#include <experimental/filesystem>
#include <fstream>
//#include <direct.h>
#include <stdexcept>
#include <chrono>
#include <string>
#include <vector>
#include <utility>
#include <random>
#include <regex>
#include <iomanip>

using namespace std;
namespace fs = std::experimental::filesystem;


typedef struct {
    bool solved;
    double elapsed_time;
    double gap;
    double obj_val;
    vector<int> *charger;
    vector<vector<float>> *yij_assigned;
    vector<double> *soc_jf_assigned;
    vector<vector<vector<float>>> *x_ijt;
    vector<double> *e_t;
} ModelState;



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
    int idxRow = 0;
    vector<string> columnNames;
    while ( getline(myFile, row)) {
        stringstream check1(row);
        string intermediate;
        int idxColumn = 0;

        while ( getline(check1, intermediate, ',')) {
            if (idxRow == 0) {
                columnNames.push_back(intermediate);
            } else {
                if(columnNames[idxColumn] == "arrivals")
                    r_j.push_back( stoi(intermediate) );
                else if(columnNames[idxColumn] == "departures")
                    d_j.push_back( stoi(intermediate) );
                else if(columnNames[idxColumn] == "soc_0")
                    soc_0.push_back( stod(intermediate) );
                else if(columnNames[idxColumn] == "battery")
                    b.push_back( stod(intermediate) );
                else if(columnNames[idxColumn] == "charger")
                    charger.push_back( stoi(intermediate) );
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


ModelState* create_model(GRBEnv const &env, double timeout, int T, int m , int n, double w, double eta, double w_G, float tau, vector<int> const&r_j, vector<int> const&d_j, vector<float>  const&soc_0, vector<float> const&b, vector<float> const&pv_t, vector<float> const&pr_t,
                         vector<vector<vector<GRBVar>>> &x_ijt, vector<GRBVar> &soc_jf, vector<vector<GRBVar>> &y_ij, vector<GRBVar> &e_t, vector<GRBVar> &u_j) {
    try {
        GRBModel model = GRBModel(env);
        stringstream name;

        float obj_coef = 0.;
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

        // Create variables x_ijt[0][0][0], ..., x_ijt[m-1][n-1][T-1]
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; ++j) {
                for(int t = 0; t < T; t++) {
                    name << "x_ijt_" << i << "-" << j << "-" << t;
                    if((t >= r_j[j]) && (t < d_j[j]))
                        x_ijt[i][j][t] = model.addVar(0, 1, obj_coef, GRB_BINARY, name.str());
                    else
                        x_ijt[i][j][t] = model.addVar(0, 0, obj_coef, GRB_BINARY, name.str());
                    name.str("");
                }
            }
        }

        // Create variables soc_jf[0], ..., soc_jf[n-1]
        for(int j = 0; j < n; j++) {
            name << "soc_" << j << "_f";
            soc_jf[j] = model.addVar(soc_0[j], 1., obj_coef, GRB_CONTINUOUS, name.str());
            name.str("");
        }

        // Create variables e_t[0], ..., e_t[T-1]
        for(int t = 0; t < T; t++) {
            name << "e_t_" << t;
            e_t[t] = model.addVar(0., w_G*tau, obj_coef, GRB_CONTINUOUS, name.str());       // Constraint (9)
            name.str("");
        }

        // Create variables y_ij[0][0], ..., y_ij[m-1][n-1]
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                name << "y_ij_" << i << "-" << j;
                y_ij[i][j] = model.addVar(0, 1, obj_coef, GRB_BINARY, name.str());
                name.str("");
            }
        }

        // Create variables u_j[0], ..., u_j[n-1]
        for(int j = 0; j < n; j++) {
            name << "u_j_" << j;
            u_j[j] = model.addVar(0, 1, obj_coef, GRB_BINARY, name.str());
            name.str("");
        }


        /////////////////////
        //// Constraints ////
        /////////////////////

        GRBLinExpr expr, expr1;
        expr.clear(); expr1.clear();

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
            }

            expr.clear();
            for(int t = r_j[j]; t < d_j[j]; t++){
                expr1.clear();
                for(int i = 0; i < m; i++){
                    expr1 += x_ijt[i][j][t];
                }
                expr += (1 - expr1);
            }
            // Constraint (2)
            for(int i = 0; i < m; i++)
                model.addConstr( expr <= (((d_j[j] - r_j[j])*(d_j[j] - r_j[j])) / (((b[j] * (.8 - soc_0[j])) / (w * eta)) * tau + (d_j[j] - r_j[j]))) * y_ij[i][j]  +  T * (1 - y_ij[i][j]) );

            for(int i = 0; i < m; i++) {
                expr1.clear();
                for(int t = 0; t < T; ++t){
                    expr1 += x_ijt[i][j][t];
                }
                model.addConstr( expr1 <= T * y_ij[i][j] );                     // Constraint (3)
            }

            for(int t = 0; t < T; ++t){
                expr.clear();
                for(int i = 0; i < m; ++i){
                    expr += x_ijt[i][j][t];
                }
                model.addConstr( expr <= 1 );                                   // Tigheting Constraint (4)
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

        ////////////////////////////
        //// Objective Function ////
        ////////////////////////////

        expr.clear();
        for(int t = 0; t < T; t++)
            expr -= e_t[t] * pr_t[t];
        for(int j = 0; j < n; ++j) {
            expr1.clear();
            for(int i = 0; i < m; i++) {
                expr1 += y_ij[i][j];
            }
            expr += expr1 * 10 + u_j[j] * 100;
        }

        model.setObjective(expr, GRB_MAXIMIZE);

        /////////////////
        //// Solving ////
        /////////////////

        model.set(GRB_DoubleParam_TimeLimit, timeout);
        double start_time = model.get(GRB_DoubleAttr_Runtime);
        model.optimize();
        double elapsed_time = model.get(GRB_DoubleAttr_Runtime) - start_time;
        int status = model.get(GRB_IntAttr_Status);
        bool solved = (status != GRB_INFEASIBLE);

        if (status == GRB_OPTIMAL) {
            cout << "Optimal solution found." << endl;
        } else if (status == GRB_TIME_LIMIT && model.get(GRB_IntAttr_SolCount) > 0) {
            cout << "Solution found within the time limit." << endl;
        } else {
            solved = false;
            cerr << "No feasible solution found." << endl;
        }


        //////////////////////////
        //// Display Solution ////
        //////////////////////////

        ModelState *model_state = (ModelState*)malloc(sizeof(ModelState));
        model_state->elapsed_time = elapsed_time;
        model_state->solved = solved;
        model_state->gap = model.get(GRB_DoubleAttr_MIPGap);
        model_state->obj_val = model.getObjective().getValue();
        model_state->yij_assigned = new vector<vector<float>>(m);
        model_state->soc_jf_assigned = new vector<double>(n);
        model_state->x_ijt = new vector<vector<vector<float>>>(m);
        model_state->charger = new vector<int>(n);
        model_state->e_t = new vector<double>(T);

        if(solved){
            cout << "soc_jf: ";
            for(int j = 0; j < n; ++j){
                (*model_state->soc_jf_assigned)[j] = soc_jf[j].get(GRB_DoubleAttr_X);
                cout << (*model_state->soc_jf_assigned)[j];
                if(j < n-1)
                    cout << ", ";
            }
            cout << endl;

            cout << "y_ij: " << endl;
            for(int i = 0; i < m; i++) {
                for(int j = 0; j < n; j++) {
                    cout << y_ij[i][j].get(GRB_DoubleAttr_X);
                    if(j < n-1)
                        cout << ", ";
                }
                cout << endl;
            }

            cout << "e_t: ";
            for(int t = 0; t < T; t++) {
                (*model_state->e_t)[t] = e_t[t].get(GRB_DoubleAttr_X);
                cout << (*model_state->e_t)[t];
                if(t < T-1)
                    cout << ", ";
            }
            cout << endl;

            double total_price = 0.;
            for(int t = 0; t < T; t++)
                total_price += (*model_state->e_t)[t] * pr_t[t];
            cout << " Total Price = " << total_price << endl;

            for(int i = 0; i < m; i++){
                (*model_state->yij_assigned).push_back(*new vector<float>(n));
                for(int j = 0; j < n; ++j){
                    (*model_state->yij_assigned)[i].push_back(0);
                }
            }

            vector<int> charger;
            for(int j = 0; j < n; j++)
                charger.push_back(0);

            for(int i = 0; i < m; i++) {
                for(int j = 0; j < n; j++) {
                    if(int(round(y_ij[i][j].get(GRB_DoubleAttr_X))) == 1) {
                        (*model_state->yij_assigned)[i][j] = 1;
                        charger[j] = i;
                    }
                }
            }

            for(int i = 0; i < m; i++) {
                model_state->x_ijt->push_back(*new vector<vector<float>>(n));
                for (int j = 0; j < n; ++j) {
                    (*model_state->x_ijt)[i].push_back(*new vector<float>(T));
                    for (int t = 0; t < T; t++) {
                        (*model_state->x_ijt)[i][j][t] = x_ijt[i][j][t].get(GRB_DoubleAttr_X);
                    }
                }
            }
        } else {
            cerr << "\tStatus: " << model.get(GRB_IntAttr_Status) << "\n";
        }

        return model_state;

    } catch(const GRBException& e) {
        cerr << "\n\nGUROBI Raised an exception:\n";
        cerr << e.getMessage() << "\n";
        throw;
    }
}


void write_log_file(ModelState *model_state, string output_filename, double timeout, int n, int m, int T, double w, double eta, double w_G, float tau,
                    vector<int> const&r_j, vector<int> const&d_j, vector<float>  const&soc_0, vector<float> const&b, vector<float> const&pv, vector<float> const&pr){
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
    ofs << "X_ijt:\n";
    for(int i = 0; i < m; i++){
        ofs << i << ":\n";
        for(int j = 0; j < n; j++){
            string list_ev = "";
            ofs << j << ":";
            for(int t = 0; t < T; t++){
                list_ev += to_string(int(round((*model_state->x_ijt)[i][j][t]*(*model_state->yij_assigned)[i][j]))) + ",";
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
            if((*model_state->yij_assigned)[i][j] == 1)
                list_ev += to_string(j) + ",";
        }
        if(list_ev.length() > 0)
            list_ev.pop_back();
        ofs << list_ev << "\n";
    }

    string list_socf = "SOC_f=";
    for(int j = 0; j < n; j++){
        list_socf += to_string((*model_state->soc_jf_assigned)[j]) + ",";
    }
    list_socf.pop_back();
    ofs << list_socf << "\n";

    string list_gt = "e_t=";
    for(int t = 0; t < T; t++){
        list_gt += to_string((*model_state->e_t)[t]) + ",";
    }
    list_gt.pop_back();
    ofs << list_gt << "\n";

    bool solved_log = model_state->solved && (model_state->elapsed_time < timeout);

    ofs << "Additional infos: \n";
    ofs << "GAP=" << to_string(model_state->gap) << "\n";
    ofs << "computation_time=" << to_string(model_state->elapsed_time) << "\n";
    ofs << "obj_val=" << to_string(model_state->obj_val) << "\n";
    ofs << "solved=" << to_string(solved_log) << "\n";

    ofs.close();
}


void run_model(int nb_ev, int ev_scenario, int pr_scenario, int pv_scenario, int station_scenario, bool use_charger_assignement, string output_filename){
    GRBEnv env = GRBEnv();
    env.start();

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
        combined.push_back(std::make_tuple(r_j[j], d_j[j], soc_0[j]));
    }
    // Sort the combined list based on the first element of each tuple
    std::sort(combined.begin(), combined.end());
    for (size_t j = 0; j < combined.size(); ++j) {
        r_j[j] = std::get<0>(combined[j]);
        d_j[j] = std::get<1>(combined[j]);
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
             << "\t" << setw(sizeof("Initial SOC")-1) <<   right << soc_0[j];
        if(charger.empty())
            cout << "\t\t" << setw(sizeof("Associated charger")-1) <<   right << -1 <<"\n";
        else
            cout << "\t\t" << setw(sizeof("Associated charger")-1) <<   right << charger[j] <<"\n";
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
    vector<GRBVar> soc_jf(n);
    vector<GRBVar> e_t(T);
    vector<vector<vector<GRBVar>>> x_ijt(m, vector<vector<GRBVar>>(n, vector<GRBVar>(T)));
    vector<vector<GRBVar>> y_ij(m, vector<GRBVar>(n));
    vector<GRBVar> u_j(n);

    if(!use_charger_assignement)
        charger.clear();
    int timeout = 30*60; // 30 minutes for offline solver

    try {
        ModelState *model_state = create_model(env, timeout, T, m, n, w, eta, w_G, tau, r_j, d_j, soc_0, b, pv, pr, x_ijt, soc_jf, y_ij, e_t, u_j);

        if (model_state->solved){
            // If CPLEX successfully solved the model, print the results
            cout << "\tSolution found!" << endl;

            write_log_file(model_state, output_filename, timeout, n, m, T, w, eta, w_G, tau, r_j, d_j, soc_0, b, pv, pr);

            // Destroy the model state
            delete model_state->yij_assigned;
            delete model_state->soc_jf_assigned;
            delete model_state->x_ijt;
            delete model_state->charger;
            delete model_state->e_t;
            free(model_state);
        } else {
            cerr << "\n\nGurobi error!\n";
        }
    } catch(const GRBException& e) {
        cerr << "\n\nGUROBI Raised an exception:\n";
        cerr << e.getMessage() << "\n";
        throw;
    }

    for(int i = 0; i < m; i++){
        x_ijt[i].end();
    }
    x_ijt.end();
    for(int i = 0; i < m; i++){
        y_ij[i].end();
    }
    y_ij.end();
    soc_jf.end();
    e_t.end();
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

