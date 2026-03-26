#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string>
#include <fstream>

class Ising
{
public:
    int N;
    double temp;  
    double total_energy;
    double magnetization;
    bool first_dump_write=true;
    int step;
    int* config;
    // Definintions for dump
    std::string file_name = "dump.ising_config";
    std::ofstream fp;
    std::string columns ="id type x y z s";
    // Class constructor
    Ising(int lattice_size, double boltzmann_temp){
        N = lattice_size;
        temp = boltzmann_temp;
        config  = new int[N*N];
    }
    ~Ising (){
        delete[] config;
        fp.close();
    }
    // void Read_input(){

    // }

    void Initialize(){
        int row,col;
        for ( row=0; row<N; row++){
            for ( col=0; col<N; col++){
                config[row*N+col] = (rand() % 2 == 0) ? -1 : 1;
                // printf("x= %d y= %d: spin=%d ", i, j, config[i][j]);
            }
        }
    }
    void MC_Move(){ 
        int row, col, spin, ry, rx, neigh_sum, cost;
        for (row=0; row<N; row++){
            for (col=0; col<N; col++){
                ry=  rand()%N;
                rx = rand()%N;
                spin = config[ry*N+rx];
                neigh_sum = config[((ry-1+N)%N)*N+rx] + config[N*((ry+1)%N)+rx] + config[ry*N+(rx-1+N)%N] + config[ry*N+(rx+1)%N];
                double rn = rand()/(double)RAND_MAX;
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
        total_energy/=4;
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
        if (first_dump_write){
            fp.open(file_name, std::ios::out);
            first_dump_write=false;
        }
        else {
            fp.open(file_name, std::ios::app);
        }

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
        fp.close();
    }

};




int main(int argn, char* argv[]){
    // std::string input_file = argv[1];
    srand(100);
    double energy, mag;
    Ising*  ising_sim = new Ising(400, 2.5);
    int nsteps = 40000;
    ising_sim->Initialize();
    ising_sim->Dump(); 
    for (int step=0; step<nsteps; step++){
        ising_sim->MC_Move();
        if (step%500==0) {
            ising_sim->Dump();
            energy = ising_sim->Calculate_energy();
            mag = ising_sim->Calculate_magnetization();
            printf("Step: %d ; Energy: %f ; Mag: %f \n", step, energy, mag);
        }
    }
    return 0;
}

