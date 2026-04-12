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
#include <cstdlib>
#include <chrono>

class Ising
{
private:
    std::mt19937 gen;    // Defining a random number generator
public:
    // Variable definition to run the Ising simulation
    int N, nsteps;       
    float temp;  
    float total_energy;
    float magnetization;
    int step;
    int* config;
    int* initial_rand_nums;
    float* rand_nums;
    int base_seed;

    // Variable definintions for dump and log outputs
    std::string dump_file_name = "dump.ising_config";
    std::string log_file_name = "log.ising";
    int output_freq;
    std::ofstream fp;
    std::ofstream logfp;
    std::string columns ="id type x y z s";



    // Class constructor  --> Uses the input file provided by the user
    Ising(std::string input_file_name){
        Read_input_file(input_file_name);
        Setup_simulation();
    }
    // Class Destructor  --> Cleans up variables and closes output files
    ~Ising (){
        fp.close();
        logfp.close();
        delete[] config;
    }
    void Read_input_file(std::string input_file_name){
        /*
            Reading the input parameters provided by the user
            First it splits them with the "=" character
            Strips them of any blank spaces
            Then convert them to the key and value pairs
            Finally the values are set to the simulation parameters defined.
        */
        std::ifstream input_file;
        try{
            input_file.open(input_file_name);
            if (!input_file.is_open()){
                throw std::runtime_error("Unable to open the input file.");
            }
            std::vector<std::string>  input_commands;
            std::string current_line;
            // Copying the input commands to a vector of strings
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
                    base_seed = std::stoi(value);
                }

            }
            input_file.close();
        }
        catch (const std::exception&e){
            std::cerr <<"Error: "<< e.what()<<std::endl;
            std::exit(EXIT_FAILURE);
        }   
        
    }

    void Setup_simulation(){
        /*
            Using parameters provided, this method sets up the variables for the simulation
        */
        config  = new int[N*N];             // Allocates heap memory on 
        initial_rand_nums = new int[N*N];
        rand_nums = new float[N*N];
        gen.seed(base_seed);                     // Sets the random seed based on the user provided value
        fp.open(dump_file_name, std::ios::out);
        logfp.open(log_file_name, std::ios::out);

    }
    
    void Initialize(){
        /*
            Initializing the Spin using random numbers --> Simulating higher temperature
            Loops over all the lattice sites and assign spin values randomly.
        */
        step=0;
        std::uniform_int_distribution<int>spin_distr(0,1);
        for (int i=0; i<N*N; i++) initial_rand_nums[i] = spin_distr(gen);
        int row,col;
        for ( row=0; row<N; row++){
            for ( col=0; col<N; col++){
                config[row*N+col] = (initial_rand_nums[row*N+col] == 0) ? -1 : 1;
            }
        }
        Calculate_magnetization();
        std::cout<<"Initial magnetization: "<<magnetization<<" \n";
    }
    void Run(){
        /*
            This function runs the simulation from step 0 to nsteps  (defined by the user input)
            It first initializes the lattice spins over the whole domain.
            Every output_frequency, output is logged in the dump file and log file, along with console print.
        */
        Initialize();
        printf("Initilized configs\n");
        // for (step=0; step<nsteps; step++){
        while (step<nsteps){
            MC_Move();
            step++;
            if (step%output_freq==0|| step==nsteps-1) {
                Calculate_energy();
                Calculate_magnetization();
                Dump();             // Writing Dump file
                Log();              // Writing Logfile
                Print_progress();   // Printing progress to the console
            }
        }
    }
    
    void MC_Move(){ 
        /*
        This function loops over all the lattice sites and attempts to flip the spin of that site
        If the energy-difference (ediff) is negative, the flipping action is automatically accepted.
        But if it is positive, it uses a random number to decide to flip it or not.

        Checkerboard update --> Consistent with the openmp implementation
        */
        std::uniform_real_distribution<float> real_distr(0.0, 1.0);
        for (int i=0; i<N*N; i++) rand_nums[i] = real_distr(gen);
        int phase, row, col, spin, neigh_sum, start_idx;
        float rn, ediff;

        // Phase 1 loop
        for (phase=0; phase<2; phase++){
            for (row=0; row<N; row++){
                // phase = 0;
                start_idx = (row+phase)%2;
                for(col=start_idx; col<N; col+=2){
                    spin = config[row*N+col];
                    neigh_sum = config[((row-1+N)%N)*N+col] + 
                                config[N*((row+1)%N)+col] + 
                                config[row*N+(col-1+N)%N] + 
                                config[row*N+(col+1)%N];
                    // rn = real_distr(gen);
                    rn = rand_nums[row*N+col];
                    ediff = static_cast<float>(2*spin*neigh_sum);
                    if (ediff<0|| rn< exp(-ediff/temp)){
                        spin*=-1;
                    }
                    config[row*N+col]=spin;
                }   
            }
        }
        
    }    

    void Print_progress(){
        printf("Step: %d ; Energy: %.2f ; Mag: %.2f \n", step, total_energy, magnetization);
    }

    void Calculate_energy(){
        /*
            Loops over the whole lattice sites and calculates the Hamiltonian.
            Hamiltonian is higher if the spin of neighboring sites are similar
        */
        int row, col, spin, neigh_sum;
        total_energy=0.0;   
        for ( row=0; row<N; row++){
            for ( col=0; col<N; col++){
                spin = config[row*N+col];
                neigh_sum = config[N*((row-1+N)%N)+col] + 
                            config[N*((row+1)%N)+col] + 
                            config[row*N+(col-1+N)%N] + 
                            config[row*N + (col+1)%N];
                total_energy+= -neigh_sum*spin;
            }
        }
        total_energy/=2;
    }

    void Calculate_magnetization(){
        /*
            Loops over lattice sites and calculate the sum of the spins--> Magnetization 
        */
        int i;
        magnetization=0.0;
        for ( i=0; i<N*N; i++){
            magnetization+=config[i];
        }
    }

    void Dump(){
        /*
            Writes the coordinates and spins of the lattice sites to a dump file
            It can be visualized in Ovito software.
        */

        // Writing the header part of the output dump file
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
    void Log(){
        /* 
            Writes progress to the log file 
        */
        if (step==0){
            logfp<<"Step       Energy        Magnetization \n";
        }
        else{
            logfp<<step<< "     "<<total_energy<<"      "<<magnetization<<"\n";
        }
        logfp.flush();
    }


};

int main(int argn, char* argv[]){
    if (argn!=2){
        std::cerr<<"Input file required\n";
        return 1;
    }
    std::string input_file = argv[1];
    float energy, mag;
    // Creating a simulation object
    Ising*  ising_sim = new Ising(input_file);
    auto start_time = std::chrono::high_resolution_clock::now();
    ising_sim->Run();     // Running the simulation
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout<<"Time taken: "<<duration.count()<<" seconds\n";

    delete ising_sim;     
    return 0;
}

