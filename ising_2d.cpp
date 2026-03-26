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
    double beta;  // Beta = kB*T  -> Boltzmann's temperature
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
    Ising(int lattice_size, int boltzmann_temp){
        N = lattice_size;
        beta = boltzmann_temp;
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
        int row, col, spin, rx, ry, neigh_sum, cost;
        for (row=0; row<N; row++){
            for (col=0; col<N; col++){
                rx = rand()%N;
                ry=  rand()%N;
                spin = config[rx*N+ry];
                neigh_sum = config[((rx-1+N)%N)*N+ry] + config[N*((rx+1)%N)+ry] + config[rx*N+(ry-1+N)%N] + config[rx*N+(ry+1)%N];
                cost = 2*spin*neigh_sum;
                if (cost<0){
                    spin*=-1;
                }
                else if ((rand()/RAND_MAX)< exp(-cost*beta)){
                    spin*=-1;
                }
                config[rx*N+ry]=spin;
            }   
        }
        step++;
    }
    void Calculate_energy(){
        int row, col, spin, neigh_sum;
        total_energy=0.0;   
        for ( row=0; row<N; row++){
            for ( col=0; col<N; col++){
                spin = config[row*N+col];
                neigh_sum = config[N*((row-1+N)%N)+col] + config[(row+1)%N+col] + config[row*N+(col-1+N)%N] + config[row*N + (col+1)%N];
                total_energy+= -neigh_sum*spin;
            }
        }
        total_energy/=4;
    }
    void Calculate_magnetization(){
        int i;
        magnetization=0.0;
        for ( i=0; i<N*N; i++){
            magnetization+=config[i];
        }
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
    Ising*  ising_sim = new Ising(100, 0.00259);
    int nsteps = 1000;
    ising_sim->Initialize();
    ising_sim->Dump();
    for (int step=0; step<nsteps; step++){
        ising_sim->MC_Move();
        ising_sim->Dump();
    }
    return 0;
}

