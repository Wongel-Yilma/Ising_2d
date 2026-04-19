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
#include <mpi.h>
#include "gpu_ising.h"



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
    float local_energy;
    float local_magnetization;
    int step;
    int local_N ;
    int * local_config;
    int* config;
    int* global_initial_rand_nums;
    int* local_initial_rand_nums;
    float* local_rand_nums;
    float* global_rand_nums;
    int base_seed;
    int rank; 
    int size;
    int up, down;
    int * upper_ghost;
    int * lower_ghost;
    IsingGPU* gpu = NULL;

    // Variable definintions for dump and log outputs
    std::string dump_file_name = "dump.ising_config";
    std::string log_file_name = "log.ising";
    int output_freq;
    std::ofstream fp;
    std::ofstream logfp;
    std::string columns ="id type x y z s";

    // Class constructor  --> Uses the input file provided by the user
    Ising(int narg, char**arg){

        std::string input_file = arg[1];
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        up    = (rank - 1 + size) % size;
        down  = (rank + 1) % size;
        if (rank==0){
            Read_input_file(input_file);
            Setup_simulation();
        }
        
        MPI_Bcast(&base_seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&temp, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nsteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&output_freq, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Assigning separate GPUs for each rank
        int numDevices;
        cudaGetDeviceCount(&numDevices);
        if(rank<numDevices){
            cudaSetDevice(rank);
        }
        else{
            std::cerr<<"Rank "<< rank<< " has no available GPU."<< std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1) ;
        }

    }
    // Class Destructor  --> Cleans up variables and closes output files
    ~Ising (){
        fp.close();
        logfp.close();
        delete[] local_config;
        delete[] local_initial_rand_nums;
        delete[] local_rand_nums;
        delete[] upper_ghost;
        delete[] lower_ghost;
        if (rank==0) {
            delete [] config;
            delete [] global_initial_rand_nums;
            delete [] global_rand_nums;
        }
        if (gpu != NULL) {
            delete gpu;
        }
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
        
        config  = new int[N*N];             // Allocates heap memory 
        global_initial_rand_nums = new int[N*N];
        global_rand_nums = new float[N*N];
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
        local_N = N/size;
        local_initial_rand_nums = new int[local_N*N];
        local_rand_nums = new float[local_N*N];
        upper_ghost = new int [N];
        lower_ghost = new int [N];
        local_config = new int [local_N*N];

        // Allocating GPU variables
        int config_size = local_N*N*sizeof(int);
        int ghost_size = N*sizeof(int);

        int rn_size = local_N*N*sizeof(float);

        if (rank==0) {
            std::uniform_int_distribution<int> spin_distr(0,1);
            for (int i=0; i<N*N; i++)   global_initial_rand_nums[i] = spin_distr(gen);
        }

        MPI_Scatter(global_initial_rand_nums, local_N*N, MPI_INT, local_initial_rand_nums, local_N*N, MPI_INT, 0, MPI_COMM_WORLD);
        
        // std::uniform_int_distribution<int>spin_distr(0,1);
        // for (int i=0; i<N*N; i++) initial_rand_nums[i] = spin_distr(gen);
        // local_rn_gen.seed(base_seed+rank);

        
        int row, col;
        for ( row=0; row<local_N; row++){
            for ( col=0; col<N; col++){
                // local_config[row*N+col] = (spin_distr(local_rn_gen) == 0) ? -1 : 1;
                local_config[row*N+col] = (local_initial_rand_nums[row*N+col] == 0) ? -1 : 1;
            }
        }

        // int local_sum = 0;
        // for (int k=0; k<N; k++) local_sum+= local_config[(local_N-1)*N+k];
        // std::cout<<"Rank "<< rank <<" ;local sum "<<local_sum<<std::endl;
       
        Comm(&local_config[0], lower_ghost, &local_config[(local_N-1)*N], upper_ghost);


        // Create a GPU object and copy the data to the GPU
        gpu = new IsingGPU(N, local_N);

        checkCuda(cudaMemcpy(gpu->input_config_d, local_config, gpu->config_size, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(gpu->output_config_d, gpu->input_config_d, gpu->config_size, cudaMemcpyDeviceToDevice));
        checkCuda(cudaMemcpy(gpu->upper_ghost_d, upper_ghost, gpu->ghost_size, cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(gpu->lower_ghost_d, lower_ghost, gpu->ghost_size, cudaMemcpyHostToDevice));

    }
    void Run(){
        /*
            This function runs the simulation from step 0 to nsteps  (defined by the user input)
            It first initializes the lattice spins over the whole domain.
            Every output_frequency, output is logged in the dump file and log file, along with console print.
        */

        // for (step=0; step<nsteps; step++){
        Initialize();
        Calculate_energy();
        Calculate_magnetization();
        if (rank==0){
            std::cout<<"Initialized configuration! "<< std::endl;
            Log();
            Print_progress();
            Dump();
        }

        while (step<nsteps){
            MC_Move( output_freq);
            step += output_freq;
            if (step%output_freq==0|| step==nsteps) {
                Calculate_energy();
                Calculate_magnetization();
                MPI_Gather(local_config, local_N*N, MPI_INT, config, local_N*N, MPI_INT, 0, MPI_COMM_WORLD);
                if (rank ==0)
                {
                    Dump();             // Writing Dump file
                    Log();              // Writing Logfile
                    Print_progress();   // Printing progress to the console
                }
            }
        }
    }
    
    void MC_Move(int steps){ 
        /*
        This function loops over all the lattice sites and attempts to flip the spin of that site
        If the energy-difference (ediff) is negative, the flipping action is automatically accepted.
        But if it is positive, it uses a random number to decide to flip it or not.

        Checkerboard update --> Consistent with the openmp implementation
        */
        checkCuda(cudaMemcpy(gpu->output_config_d, gpu->input_config_d, gpu->config_size, cudaMemcpyDeviceToDevice));
        for (int s=0; s<steps; s++){
            if (rank==0){
                std::uniform_real_distribution<float> real_distr(0.0, 1.0);
                for (int i=0; i<N*N; i++)   global_rand_nums[i] = real_distr(gen);
            }
            MPI_Scatter(global_rand_nums, local_N*N, MPI_FLOAT, local_rand_nums, local_N*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
            checkCuda(cudaMemcpy(gpu->rand_nums_d, local_rand_nums, gpu->rn_size, cudaMemcpyHostToDevice));
            /*
                Phase 0 --> White sites
            */
            gpu->launch_kernel_ising(0, temp, rank);
            checkCuda(cudaMemcpy(gpu->input_config_d, gpu->output_config_d, gpu->config_size, cudaMemcpyDeviceToDevice));
            /*
                Communicate the ghost sites
            */
            Comm(gpu->input_config_d,
                gpu->lower_ghost_d,
                gpu->input_config_d + (local_N-1)*N,
                gpu->upper_ghost_d);

            /*
                Phase 1 --> Black sites
            */
            gpu->launch_kernel_ising(1, temp, rank);
            checkCuda(cudaMemcpy(gpu->input_config_d, gpu->output_config_d, gpu->config_size, cudaMemcpyDeviceToDevice));
            /*
                Communicate the ghost sites
            */
            Comm(gpu->input_config_d,
                gpu->lower_ghost_d,
                gpu->input_config_d + (local_N-1)*N,
                gpu->upper_ghost_d);
        }
        // Once output frequency is reached, copy the data from the GPU to the host
        checkCuda(cudaMemcpy(local_config, gpu->input_config_d, gpu->config_size, cudaMemcpyDeviceToHost) ); 
    
    }    
    void Comm(int * send_buffer_1, int *recv_buffer_1, int * send_buffer_2, int *recv_buffer_2){
        /*
             Sending and receiving the boundary data
             send_buffer_1 --> Sending the upper boundary data (&local_config[0])
             recv_buffer_1 --> Receiving the lower boundary data (&lower_ghost)
             send_buffer_2 --> Sending the lower boundary data (&local_config[(local_N-1)*N])
             recv_buffer_2 --> Receiving the upper boundary data (&upper_ghost)
        */
        MPI_Sendrecv(send_buffer_1, N, MPI_INT, up, 0, recv_buffer_1, N, MPI_INT, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_buffer_2, N, MPI_INT, down, 0, recv_buffer_2, N, MPI_INT, up, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
        local_energy=0.0;   
        for ( row=0; row<local_N; row++){
            for ( col=0; col<N; col++){
                spin = local_config[row*N+col];
                neigh_sum = local_config[row*N+(col-1+N)%N] + 
                                local_config[row*N+(col+1)%N];
                if (row==0){
                    neigh_sum += upper_ghost[col] + local_config[(row+1)*N+col];
                }
                else if (row==local_N-1){
                    neigh_sum += lower_ghost[col] + local_config[(row-1)*N+col];
                }
                else{
                    neigh_sum += local_config[(row-1)*N+col]+local_config[(row+1)*N+col];
                }
                local_energy+= -neigh_sum*spin;
            }
        }
        local_energy/=2;
        MPI_Reduce(&local_energy, &total_energy, 1, MPI_FLOAT , MPI_SUM, 0, MPI_COMM_WORLD);
    }

    void Calculate_magnetization(){
        /*
            Loops over lattice sites and calculate the sum of the spins--> Magnetization 
        */
        int i;
        local_magnetization=0.0;
        for ( i=0; i<local_N*N; i++){
            local_magnetization+=local_config[i];
        }
        MPI_Reduce(&local_magnetization, &magnetization, 1, MPI_FLOAT , MPI_SUM, 0, MPI_COMM_WORLD);
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

int main(int argn, char** argv){

    MPI_Init(&argn, &argv); // This is making everything MPI
    float start, end, elapsed;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    start =  MPI_Wtime();
    // Creating a simulation object
    Ising*  ising_sim = new Ising(argn, argv);
    ising_sim->Run();

    end = MPI_Wtime();
    elapsed = end - start;

    if (rank==0){
        std::cout<< "Runtime: "<<  elapsed<< std::endl;
    }
    delete ising_sim;    
    MPI_Finalize();
}

