#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void Initialize(int , int**);
void MC_move(int ,int**, double);
void Calculate_energy(int, int**, double *);
void Calculate_magnetization(int, int**, double *);


int main(){
    int N =4;
    int ** config;
    config = malloc(N*sizeof(int*));
    for (int i=0; i<N; i++){
        config[i] = malloc(N*sizeof(int));
    }
    Initialize(N, config);
    // Declare the neccessary variablesto run the sim
    int nsteps = 10;
    double T = 1.5;
    double total_energy;
    double magnetization;
    int step;
    for (step=0; step<nsteps; step++){
        MC_move(N,  config, T);
        Calculate_energy(N,config, &total_energy);
        Calculate_magnetization(N,config, &magnetization);
        printf("Step: %d    Energy: %f    Mag: %f     \n", step, total_energy, magnetization);
    }

    return 0;
}

void Initialize(int N, int**config){
    int i,j;
    for ( i=0; i<N; i++){
        for ( j=0; j<N; j++){
            config[i][j] = (rand() % 2 == 0) ? -1 : 1;
            printf("x= %d y= %d: spin=%d ", i, j, config[i][j]);
        }
    }
}

void MC_move(int N, int**config, double beta){
    int i, j, spin, rx, ry, neigh_sum, cost;


    for (i=0; i<N; i++){
        for (j=0; j<N; j++){
            rx = rand()%N;
            ry=  rand()%N;
            spin = config[rx][ry];
            neigh_sum = config[(rx-1+N)%N][ry] + config[(rx+1)%N][ry] + config[rx][(ry-1+N)%N] + config[rx][(ry+1)%N];
            cost = 2*spin*neigh_sum;
            if (cost<0){
                spin*=-1;
            }
            else if ((rand()/RAND_MAX)< exp(-cost*beta)){
                spin*=-1;
            }
            config[rx][ry]=spin;
        }   
    }
}

void Calculate_energy(int N, int** config , double* total_energy){
    int i, j, spin, neigh_sum;
    *total_energy=0.0;
    for ( i=0; i<N; i++){
        for ( j=0; j<N; j++){
            spin = config[i][j];
            neigh_sum = config[(i-1+N)%N][j] + config[(i+1)%N][j] + config[i][(j-1+N)%N] + config[i][(j+1)%N];
            *total_energy+= -neigh_sum*spin;
        }
    }
    *total_energy/=4;
}

void Calculate_magnetization(int N, int** config , double* magnetization){
    int i, j, spin, neigh_sum;
    *magnetization=0.0;
    for ( i=0; i<N; i++){
        for ( j=0; j<N; j++){
            *magnetization+=config[i][j];
        }
    }
}