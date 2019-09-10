#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <ctime>

using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////FUNCTION DECLARATIONS///////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

//yield_curve function
double yield_curve(const double& time);
//hazard rate getter
double haz_fn(const double& time, const vector<double>& lambdas, const vector<double>& maturities);
//function for CIR bond price
double CIR_bond_price(const double& t,const double& y0,const double& kappa,const double& mu,const double& nu);
//function for PSI
double PSI(const double& t,const double& y0,const double& kappa,const double& mu,const double& nu,
           const vector<double>& lambdas, const vector<double>& maturities);
//function for the actual phi
double phi(const double& t,const double& y0,const double& kappa,const double& mu,const double& nu,
           const vector<double>& lambdas, const vector<double>& maturities);
//function for the fitted survival probabilities
double fitted_sp(const double& t,const double& y0,const double& kappa,const double& mu,const double& nu,
                 const vector<double>& lambdas, const vector<double>& maturities);
//functions for working out CVA
double CVA_MC_Call(const double& maturity, const double& correlation, const double& recovery,
                   const double& y0,const double& kappa,const double& mu,const double& nu, const vector<double>& lambdas,
                   const vector<double>& maturities , const vector<double>& phi_vals);
double Lambda_inverse(const double& exp_sim, const vector<double>& intensity, const double& interval_length,
                      const double& maturity);
double norm_cdf(const double &x);
double d_j(const int &j, const double &S, const double &T,const double &t);
double black_scholes(const double& T, const double& t, const double & S_t);

//functions for CDS bootstrap
double CDS_price(const double& spread, const double& maturity, const vector<double>& lambdas,
        const vector<double>& maturities, const double& recovery);
double pd_getter(const double& t, const vector<double>& lambdas, const vector<double>& maturities);
void optimise_CDS_price(const int& k, vector<double>& lambda_guesses, const vector<double>& maturities,
        const vector<double>& market_spreads, const double& recovery);

//functions for simulation of random numbers
void rand_uniform(double&);
void rand_exp(double&);
void rand_normal(vector<double> &);
void correlate_normals(const vector<double> &, vector<double> &, const double&);


//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////STATING GLOBAL CONSTANTS/////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

//Parameters for the option and market
double S_0 = 100, K = 100, r = 0.05, sigma = 0.40, maturity_of_call = 5.0;
//Number of simulations and number of discretisations for euler per simulation
// and number of discretisations for the intensity lambda
unsigned long int M = 1000000, nde = 1000, nd = 1000;
//Parameters for the congruential random uniform generator
unsigned long int seed = 30000000, a = 16807, m = 2147483647;

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////MAIN CODE WITH FUNCTION IMPLEMENTATIONS BELOW///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

int main() {
    //example maturities and spreads
    vector<double> maturities{0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
    vector<double> market_spreads{22*0.0001,40*0.0001,60*0.0001,85*0.0001,95*0.0001,105*0.0001};
    double recovery = 0.4;

    unsigned int n = maturities.size();

    vector<double> bond_price(n);
    for(int k=0;k<n;k++){
        bond_price[k] = exp(-maturities[k]*r);
    }

    // need the time steps
    vector<double> deltati(n-1);
    for(int i =0;i<(n-1);i++){
        deltati[i] = maturities[i+1]-maturities[i];
    }

//    //first way to calculate survival probabilities (slightly inaccurate)
    vector<double> survival_probabilities_rough;
    survival_probabilities_rough.push_back(1.0);
    double sum1 = 0.0;
    double sum2 = 0.0;
    for(int i=0; i<(n-1);i++){
        survival_probabilities_rough.push_back((sum1 + (1-recovery)*yield_curve(maturities[i+1])*
        survival_probabilities_rough[i] - market_spreads[i]*sum2)/(market_spreads[i]*yield_curve(maturities[i+1])
        *deltati[i]+(1-recovery)*yield_curve(maturities[i+1])));
        sum1 += (1-recovery)*yield_curve(maturities[i+1])*(survival_probabilities_rough[i]-
                survival_probabilities_rough[i+1]);
        sum2 += yield_curve(maturities[i+1])*survival_probabilities_rough[i+1]*deltati[i];
    }


    //For the plots
    ofstream fout("sp1.txt");
    if(! fout.is_open()){cout << "Trouble outputting to file";return -1;}
    for(int i = 0; i < survival_probabilities_rough.size(); i++){
        fout << maturities[i] << " " << survival_probabilities_rough[i] << endl;
    }
    fout.close();


    //Next we look for a way to get the implied intensities by CDS bootstrapping and then obtain the
    // relevent implied probabilities

// The below is for testing the bootstrapping. Ideally with the fitted hazard rates and the market spread quotes
// the price should be zero.
//    vector<double> tempm(2,0.0);
//    tempm[1] = 1.0;
//    cout << CDS_price(0.0026, 1.0, vector<double>(1,0.45), tempm, 0.45) << endl;

    vector<double> lambda_guesses(n-1,0);
    for(int k = 0; k < (n-1); k++){
        optimise_CDS_price(k, lambda_guesses, maturities, market_spreads, recovery);
    }

    //For the plot of intensity
    ofstream fout2("intensities.txt");
    if(! fout2.is_open()){cout << "Trouble outputting to file";return -1;}
    fout2 << maturities[0] << " " << 0.0 << endl;
    for(int i = 0; i < lambda_guesses.size(); i++){
        fout2 << maturities[i+1] << " " << lambda_guesses[i] << endl;
    }
    fout2.close();

    //For the plot of the hazard function
    vector<double> x_points(1000);
    vector<double> y_points(1000);
    double spacing = (maturities.back()-maturities[0])/1000;
    double input = 0;
    for(int k = 0; k<1000;k++){
        x_points[k] = input;
        y_points[k] = -haz_fn(input,lambda_guesses,maturities);
        input += spacing;
    }
    ofstream fout3("hazfn.txt");
    if(! fout3.is_open()){cout << "Trouble outputting to file";return -1;}
    for(int i = 0; i < x_points.size(); i++){
        fout3 << x_points[i] << " " << y_points[i] << endl;
    }
    fout3.close();

    //For the plots of survival probability second way, this time sampling from a lot more times
    // and its a lot more accurate
    vector<double> x_points2(1000);
    vector<double> y_points2(1000);
    double spacing2 = (maturities.back()-maturities[0])/1000;
    double input2 = 0;
    for(int k = 0; k<1000;k++){
        x_points2[k] = input2;
        y_points2[k] = exp(-haz_fn(input2,lambda_guesses,maturities));
        input2 += spacing2;
    }
    ofstream fout4("sp2.txt");
    if(! fout4.is_open()){cout << "Trouble outputting to file";return -1;}
    for(int i = 0; i < x_points2.size(); i++){
        fout4 << x_points2[i] << " " << y_points2[i] << endl;
    }
    fout4.close();

    //for comparison of survival probabilities if the 2 methods in
    vector<double> survival_probabilities2;
    survival_probabilities2.push_back(1.0);
    for(int i=0; i<(n-1);i++){
        survival_probabilities2.push_back(exp(-haz_fn(maturities[i+1],lambda_guesses,maturities)));
    }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//    NOTE the market implied forward curve are just the intensities
//    So now we guess parameters for CIR++ and see if our resulting phi is positive

//    lets first inspect the lambda curve as that is our implied forwards rate curve
//    for(const double& l : lambda_guesses){
//        cout << l << endl;
//    }
    //We see that it's increasing with a slight hump, we can choose our CIR++ parameters so that
    //we have our start point y0 below the start of the intensities and our end point below the limit
    double y0 = 0.003;
    double kappa = 0.01;
    double mu = 0.018;
    double nu = 0.01;
    if(2*kappa*mu <= nu*nu){
        cout << "Seems as if the condition 2*kappa*mu > nu^2 is not met" << endl;
        return 1;
    }

    //For the plotof the final phi making sure that its positive
    vector<double> x_points3(1000);
    vector<double> y_points3(1000);
    double spacing3 = (maturities.back()-maturities[0])/1000;
    double input3 = 0;
    for(int k = 0; k < 1000;k++){
        x_points3[k] = input3;
        y_points3[k] = phi(input3,y0,kappa,mu,nu,lambda_guesses,maturities);
        input3 += spacing3;
    }
    ofstream fout5("phi.txt");
    if(! fout5.is_open()){cout << "Trouble outputting to file";return -1;}
    for(int i = 0; i < x_points3.size(); i++){
        fout5 << x_points3[i] << " " << y_points3[i] << endl;
    }
    fout5.close();

    //for plotting the fitted survival probabilities against market implied probabilities
    vector<double> x_points4(1000);
    vector<double> y_points4(1000);
    double spacing4 = (maturities.back()-maturities[0])/1000;
    double input4 = 0;
    for(int k = 0; k < 1000;k++){
        x_points4[k] = input4;
        y_points4[k] = fitted_sp(input4,y0,kappa,mu,nu,lambda_guesses,maturities);
        input4 += spacing4;
    }
    ofstream fout6("fittedsp.txt");
    if(! fout6.is_open()){cout << "Trouble outputting to file";return -1;}
    for(int i = 0; i < x_points4.size(); i++){
        fout6 << x_points4[i] << " " << y_points4[i] << endl;
    }
    fout6.close();
















    //To improve the efficiency of MC we calculate the phi outside and pass it as an input as it never changes
    vector<double> x_vals(nd);
    vector<double> phi_vals(nd);
    double interval_length = (maturities.back()-maturities[0])/(nd-1);
    for(int k=0;k<nd;k++){
        x_vals[k] = k*interval_length;
        phi_vals[k] = phi(x_vals[k],y0,kappa,mu,nu,lambda_guesses,maturities);
    }


    //For plotting the CVA for different correlations
    unsigned int nr = 11;
    vector<double> rhos(nr);
    rhos[0] = -1.;
    for(int k = 1; k<nr;k++){
        rhos[k] = rhos[k-1]+2./(nr-1);
    }
    vector<double> CVAs(nr);
    for(int k = 0; k < nr;k++){
        cout << k << endl;
        CVAs[k] = CVA_MC_Call(maturity_of_call, rhos[k] , recovery, y0, kappa, mu, nu,
                lambda_guesses, maturities , phi_vals);
        cout << CVAs[k] << endl;
    }
    ofstream fout7("CVAs001_01_020.txt");
    if(! fout7.is_open()){cout << "Trouble outputting to file";return -1;}
    for(int i = 0; i < rhos.size(); i++){
        fout7 << rhos[i] << " " << CVAs[i] << endl;
    }
    fout7.close();


    return 0;
}

double yield_curve(const double& time){
    double r = 0.02;
    return exp(-r*time);
}

double CDS_price(const double& spread, const double& maturity, const vector<double>& lambdas,
        const vector<double>& maturities, const double& recovery){
    double discretisation_increment = 0.05; //corresponds to about 3 weeks

    //getting the payment dates
    vector<double> payment_dates;
    double temptime = 0.0;
    while(temptime < maturity){payment_dates.push_back(temptime);temptime += 0.25;}
    payment_dates.push_back(maturity);

    double integral = 0, t_d = 0;
    int beta_t_d = 1, counter = 1, switch_point = 5;    //NOTE! these need to be picked carefully according to the
    // dicretisation_increment and the payment dates

    for(int k = 0; k < (int)(maturity/discretisation_increment); k++){
        integral += yield_curve(t_d)*(spread*(t_d-payment_dates[beta_t_d-1])-1+recovery)*
                (pd_getter(t_d+discretisation_increment,lambdas,maturities)-pd_getter(t_d,lambdas,maturities));

        t_d += discretisation_increment;
        if(counter == switch_point){
            counter = 0;
            beta_t_d++;
        }
        counter++;
    }

    //Now for the sum
    double sum = 0;
    for(int k = 1; k < payment_dates.size();k++){
        sum += (1-pd_getter(payment_dates[k],lambdas,maturities))*yield_curve(payment_dates[k])*spread*
                (payment_dates[k]-payment_dates[k-1]);
    }
    return integral + sum;
}

double pd_getter(const double& t, const vector<double>& lambdas, const vector<double>& maturities){
    // to catch if t is outside the range of maturities
    if(t > maturities.back()){return pd_getter(maturities.back(),lambdas,maturities) +
    (t-maturities.back())*lambdas.back();}

    int i = 0;
    double integral = 0;
    while(t > maturities[i+1]){
        integral += (maturities[i+1]-maturities[i])*lambdas[i];
        i++;
    }
    integral += (t-maturities[i])*lambdas[i];
    return 1 - exp(-integral);
}

void optimise_CDS_price(const int& k, vector<double>& lambda_guesses, const vector<double>& maturities,
        const vector<double>& market_spreads, const double& recovery){
    //this uses interval bisection to find the lambda that will get the CDS price to 0
    double curr_spread = market_spreads[k];
    double maturity = maturities[k+1];

    double left = -0.1;
    double right = 10.0;
    double centre = (left+right)/2;
    double tolerance = 0.0001;

    int c = 0;
    double guess = 1.0;
    while(abs(guess) > tolerance){
        lambda_guesses[k] = centre;
        guess = CDS_price(curr_spread, maturity, lambda_guesses, maturities, recovery);
        if(guess < 0){
            right = centre;
            centre = (left+right)/2;
        }
        else {
            left = centre;
            centre = (left+right)/2;
        }
    }
}

double haz_fn(const double& time, const vector<double>& lambdas, const vector<double>& maturities){
    //now we create a function for the implied hazard rate
    if(time > maturities.back()){
        return haz_fn(maturities.back(),lambdas,maturities)+(time-maturities.back())*lambdas.back();
    }

    int curr = 0;
    double haz = 0;
    while(maturities[curr+1] < time){
        haz += (maturities[curr+1]-maturities[curr])*lambdas[curr];
        curr += 1;
    }
    haz += (time-maturities[curr])*lambdas[curr];
    return haz;
}

double CIR_bond_price(const double& t,const double& y0,const double& kappa,const double& mu,const double& nu){
    double h = sqrt(kappa*kappa + 2*y0*y0);
    double a = pow(( (2*h*exp((h+kappa)*t/2))/(2*h+(h+kappa)*(exp(h*t)-1)) ),(2*kappa*mu/(nu*nu)));
    double b = (2*(exp(h*t)-1))/(2*h+(h+kappa)*(exp(h*t)-1));
    return a*exp(-y0*b);
}

double PSI(const double& t,const double& y0,const double& kappa,const double& mu,const double& nu,
        const vector<double>& lambdas, const vector<double>& maturities){
    return haz_fn(t,lambdas,maturities) + log(CIR_bond_price(t,y0,kappa,mu,nu));
}

double phi(const double& t,const double& y0,const double& kappa,const double& mu,const double& nu,
           const vector<double>& lambdas, const vector<double>& maturities){
    double epsilon2 = 0.01;
    return (PSI(t+epsilon2,y0,kappa,mu,nu,lambdas,maturities)-PSI(t-epsilon2,y0,kappa,mu,nu,lambdas,maturities))
    /(2*epsilon2);
}

double fitted_sp(const double& t,const double& y0,const double& kappa,const double& mu,const double& nu,
                 const vector<double>& lambdas, const vector<double>& maturities){
    return exp(-PSI(t,y0,kappa,mu,nu,lambdas,maturities))*CIR_bond_price(t,y0,kappa,mu,nu);
}

void rand_uniform(double &i) {
    //Congruential Random uniform generator
    seed = (a * seed) % m;
    i = (double) seed / m;
}

void rand_exp(double &i){
    rand_uniform(i);
    i = -log(i);
}

void rand_normal(vector<double> &input) {
    //Marsaglia's Polar Method
    double unif1, unif2, v1, v2, w, sqf;
    for (int k = 0; k < (floor(input.size() / 2)); k++) {
        do {
            rand_uniform(unif1);
            rand_uniform(unif2);
            v1 = 2 * unif1 - 1;
            v2 = 2 * unif2 - 1;
            w = v1 * v1 + v2 * v2;
        } while (w > 1);
        sqf = sqrt((-2 * log(w)) / (w));
        input[2 * k] = sqf * v1;
        input[2 * k + 1] = sqf * v2;
    }
    if (input.size() % 2) {
        do {
            rand_uniform(unif1);
            rand_uniform(unif2);
            v1 = 2 * unif1 - 1;
            v2 = 2 * unif2 - 1;
            w = v1 * v1 + v2 * v2;
        } while (w > 1);
        input[input.size() - 1] = sqrt((-2 * log(w)) / (w)) * v1;
    }
}

void correlate_normals(const vector<double> &n1, vector<double> &n2, const double& rho) {
    for (int k = 0; k < n2.size(); k++) {
        n2[k] = rho * n1[k] + sqrt(1 - rho * rho) * n2[k];
    }
}

double Lambda_inverse(const double& exp_sim, const vector<double>& intensity, const double& interval_length,
        const double& maturity){
    double sumi = 0.;
    int index = 0;
    double in_size = intensity.size();

    while(exp_sim > sumi && index < in_size){
        sumi += intensity[index]*interval_length;
        index++;
    }
    if(exp_sim > sumi){
        return 2*maturity; //basically make sure you return a time above maturity
    }

    return index*interval_length;
}

double norm_cdf(const double &x) {
    return 0.5 * erfc(-x * M_SQRT1_2);
}

double d_j(const int &j, const double &S, const double &T,const double &t) {
    return (log(S / K) + (r + (pow(-1, j - 1)) * 0.5 * sigma * sigma) * (T-t)) / (sigma * (pow(T-t, 0.5)));
}

double black_scholes(const double& T, const double& t, const double & S_t){
    return S_t * norm_cdf(d_j(1, S_t, T,t)) - K * exp(-r * (T-t)) * norm_cdf(d_j(2, S_t, T,t));
}

double CVA_MC_Call(const double& maturity, const double& correlation, const double& recovery,
        const double& y0,const double& kappa,const double& mu,const double& nu, const vector<double>& lambdas,
        const vector<double>& maturities , const vector<double>& phi_vals) {

    double sum = 0;

    vector<double> intensity_normals(nd);
    vector<double> intensity(nd);
    vector<double> stochastic_part(nd);
    stochastic_part[0] = y0;
    intensity[0] = y0 + phi_vals[0];
    double h = (maturities.back()-maturities[0])/(nd-1);
    double exp_sim;
    double default_time;
    unsigned int default_index;
    for(int k=0; k<M; k++){
        //first we generate an intensity path
        rand_normal(intensity_normals);
        for(int j=1;j<nd;j++){
            stochastic_part[j] = stochastic_part[j-1] + (kappa*(mu-stochastic_part[j-1]))*h
                    + (nu*sqrt(stochastic_part[j-1]))*sqrt(h)*intensity_normals[j];
            intensity[j] = stochastic_part[j] + phi_vals[j];
        }
        //Once this path has been generated now we want to simulate a default time so first simulate an exponential
        rand_exp(exp_sim);
        default_time = Lambda_inverse(exp_sim,intensity,h,maturity);
        if(default_time >= maturity){continue;}


        //Now we need to simulate the stock path up until the default time
        default_index = static_cast<unsigned int>(default_time / h);
        vector<double> stock_normals(default_index);
        rand_normal(stock_normals);
        correlate_normals(intensity_normals,stock_normals,correlation);

        vector<double> stock_path(default_index);
        stock_path[0] = S_0;
        for(int j=1; j<default_index; j++){

            stock_path[j] = stock_path[j-1] + (r*stock_path[j-1])*h
                                 + (sigma*stock_path[j-1])*sqrt(h)*stock_normals[j];
        }

        sum += yield_curve(default_time)*black_scholes(maturity,default_time,stock_path[default_index-1]);
    }


    double expectation = sum/M;
    return (1-recovery)*expectation;
}

