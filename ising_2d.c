#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void Initialize(int , int**);
void MC_move(int ,int**, double);

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
    MC_move(N,  config, T);



    return 0;
}

void Initialize(int N, int**config){
    int i,j;
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            config[i][j] = (int)( 2*((double)rand()/RAND_MAX)-1);
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
            neigh_sum = config[(rx-1)%N][ry] + config[(rx+1)%N][ry] + config[rx][(ry-1)%N] + config[rx][(ry+1)%N];
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