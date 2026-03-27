#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <random>

class Ising
{
private:
    std::mt19937 gen;
public:
    int N, nsteps;
    double temp;  
    double total_energy;
    double magnetization;
    int step;
    int* config;
    int seed;
    // Definintions for dump and log outputs
    std::string dump_file_name = "dump.ising_config";
    std::string log_file_name = "log.ising";
    int output_freq;
    std::ofstream fp;
    std::string columns ="id type x y z s";

    // Class constructor
    Ising(std::string input_file_name){
        Read_input_file(input_file_name);
        Setup_simulation();
    }
    ~Ising (){
        fp.close();
        delete[] config;
    }
    void Read_input_file(std::string input_file_name){
        std::ifstream input_file;
        try{
            input_file.open(input_file_name);
            if (!input_file.is_open()){
                throw std::runtime_error("Unable to open the input file.");
            }
            std::vector<std::string>  input_commands;
            std::string current_line;
            while(std::getline(input_file,current_line)){
                input_commands.push_back(current_line);
            }
            input_file.close();
            //  Parsing the input commands as a Key and value pair
            std::string key, value;

            for (int j=0; j<input_commands.size(); j++){
                size_t pos = input_commands[j].find("=");
                key = input_commands[j].substr(0, pos);
                value = input_commands[j].substr(pos+1);
                key.erase(std::remove_if(key.begin(), key.end(), ::isspace), key.end());
                value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());    
                std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c){return std::tolower(c);});
                if (key=="n"){
                    N = std::stoi(value);
                }
                else if(key=="t"){
                    temp = std::stof(value);
                }
                else if(key=="nsteps"){
                    nsteps = std::stoi(value);
                }
                else if (key=="output_file"){
                    dump_file_name = value;
                }
                else if (key=="output_frequency"){
                    output_freq = std::stoi(value);
                }
                else if (key=="seed"){
                    seed = std::stoi(value);
                }

            }
            input_file.close();
        }
        catch (const std::exception&e){
            std::cerr <<"Error: "<< e.what()<<std::endl;
        }   
        
    }
    void Setup_simulation(){
        config  = new int[N*N];
        gen.seed(seed);
        fp.open(dump_file_name, std::ios::out);

    }

    void Initialize(){
        step=0;
        std::uniform_int_distribution<int>spin_distr(0,1);
        int row,col;
        for ( row=0; row<N; row++){
            for ( col=0; col<N; col++){
                config[row*N+col] = (spin_distr(gen) == 0) ? -1 : 1;
                // printf("x= %d y= %d: spin=%d ", i, j, config[i][j]);
            }
        }
    }
    void MC_Move(){ 
        std::uniform_int_distribution<int> int_distr(0,N-1);
        std::uniform_real_distribution<double> real_distr(0.0, 1.0);
        int row, col, spin, ry, rx, neigh_sum, cost;
        double rn;
        for (row=0; row<N; row++){
            for (col=0; col<N; col++){
                // ry=  rand()%N;
                // rx = rand()%N;
                ry = int_distr(gen);
                rx = int_distr(gen);
                spin = config[ry*N+rx];
                neigh_sum = config[((ry-1+N)%N)*N+rx] + config[N*((ry+1)%N)+rx] + config[ry*N+(rx-1+N)%N] + config[ry*N+(rx+1)%N];
                rn = real_distr(gen);
                cost = 2*spin*neigh_sum;
                if (cost<0){
                    spin*=-1;
                }
                else if (rn< exp(-cost/temp)){
                    spin*=-1;
                }
                config[ry*N+rx]=spin;
            }   
        }
        step++;
    }
    double Calculate_energy(){
        int row, col, spin, neigh_sum;
        total_energy=0.0;   
        for ( row=0; row<N; row++){
            for ( col=0; col<N; col++){
                spin = config[row*N+col];
                neigh_sum = config[N*((row-1+N)%N)+col] + config[N*((row+1)%N)+col] + config[row*N+(col-1+N)%N] + config[row*N + (col+1)%N];
                total_energy+= -neigh_sum*spin;
            }
        }
        total_energy/=2;
        return total_energy;
    }
    double Calculate_magnetization(){
        int i;
        magnetization=0.0;
        for ( i=0; i<N*N; i++){
            magnetization+=config[i];
        }
        return magnetization;
    }

    void Dump(){
        // Writing the header
        fp<<"ITEM: TIMESTEP\n";
        fp<<step<<" "<<step<<"\n";
        fp<<"ITEM: NUMBER OF ATOMS\n";
        fp<<N*N<<"\n";
        fp<<"ITEM: BOX BOUNDS\n";
        fp<<0<< " "<<N<<"\n";
        fp<<0<< " "<<N<<"\n";
        fp<<-0.5<< " "<<0.5<<"\n";
        fp<<"ITEM: ATOMS "<<columns<<"\n";
        // Write the position and spin data
        int i;
        for (i=0; i<N*N;i++){
            fp<< i<<" "<<1<<" "<<i/N<<" "<<i%N<<" "<<0<<" "<<config[i]<<"\n";
        }
        fp.flush();
        
    }

};




int main(int argn, char* argv[]){
    if (argn!=2){
        std::cerr<<"Input file required";
        return 1;
    }
    std::string input_file = argv[1];
    double energy, mag;
    Ising*  ising_sim = new Ising(input_file);
    int nsteps = 400000;
    ising_sim->Initialize();
    ising_sim->Dump(); 
    for (int t=0; t<nsteps; t++){
        ising_sim->MC_Move();
        if (t%5000==0) {
            ising_sim->Dump();
            energy = ising_sim->Calculate_energy();
            mag = ising_sim->Calculate_magnetization();
            printf("Step: %d ; Energy: %f ; Mag: %f \n", t, energy, mag);
        }
    }
    delete ising_sim;
    return 0;
}

