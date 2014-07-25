/***************************************************************************
 *                           BDF1vs.cpp                                    *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Francisco Naveros                    *
 * email                : fnaveros@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/integration_method/BDF1vs.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"

#include <math.h>


BDF1vs::BDF1vs(TimeDrivenNeuronModel * NewModel, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState):VariableStep(NewModel, "BDF1vs", N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState, true, true){
}

BDF1vs::~BDF1vs(){
	delete [] D;
	delete [] OriginalD;
	delete [] State;
	delete [] OriginalState;
}
		
void BDF1vs::NextDifferentialEcuationValue(int index, float * NeuronState, float elapsed_time){
	float AuxNeuronState[MAX_VARIABLES];
	float AuxNeuronState_p[MAX_VARIABLES];
	float AuxNeuronState_p1[MAX_VARIABLES];
	float AuxNeuronState_c[MAX_VARIABLES];
	float jacnum[MAX_VARIABLES*MAX_VARIABLES];
	float J[MAX_VARIABLES*MAX_VARIABLES];
	float inv_J[MAX_VARIABLES*MAX_VARIABLES];

	memcpy(OriginalD + (index*N_DifferentialNeuronState), D + (index*N_DifferentialNeuronState), N_DifferentialNeuronState*sizeof(float));
	OriginalState[index]=State[index];


	if(State[index]==0){
		model->EvaluateDifferentialEcuation(NeuronState, AuxNeuronState);
		for (int j=0; j<N_DifferentialNeuronState; j++){
			AuxNeuronState_p[j]= NeuronState[j] + elapsed_time*AuxNeuronState[j];
		}
	}else{
		for (int j=0; j<N_DifferentialNeuronState; j++){
			AuxNeuronState_p[j]= NeuronState[j] + elapsed_time*D[index*N_DifferentialNeuronState+j];
		}
	}

	//memcpy(offset_AuxNeuronState_p + N_DifferentialNeuronState, NeuronState + N_DifferentialNeuronState ,sizeof(float)* (N_NeuronStateVariables-N_DifferentialNeuronState));
	for(int i=N_DifferentialNeuronState; i<N_NeuronStateVariables; i++){
		AuxNeuronState_p[i]=NeuronState[i];
	}
		
	model->EvaluateTimeDependentEcuation(AuxNeuronState_p,elapsed_time);

	float epsilon=1.0;
	int k=0;

	while (epsilon>1e-16 && k<5){
		model->EvaluateDifferentialEcuation(AuxNeuronState_p, AuxNeuronState);
		for (int j=0; j<N_DifferentialNeuronState; j++){
			AuxNeuronState_c[j]=NeuronState[j] + elapsed_time*AuxNeuronState[j];
		}

		//jacobian.
		Jacobian(AuxNeuronState_p, jacnum, elapsed_time);
	
		for(int z=0; z<N_DifferentialNeuronState; z++){
			for(int t=0; t<N_DifferentialNeuronState; t++){
				J[z*N_DifferentialNeuronState + t] = elapsed_time * jacnum[z*N_DifferentialNeuronState + t];
				if(z==t){
					J[z*N_DifferentialNeuronState + t]-=1;
				}
			}
		}
		this->invermat(J,inv_J);
		for(int z=0; z<N_DifferentialNeuronState; z++){
			float aux=0.0;
			for (int t=0; t<N_DifferentialNeuronState; t++){
				aux+=inv_J[z*N_DifferentialNeuronState+t]*(AuxNeuronState_p[t]-AuxNeuronState_c[t]);
			}
			AuxNeuronState_p1[z]=aux + AuxNeuronState_p[z];
		}

		float aux=0.0;
		float aux2=0.0;
		for(int z=0; z<N_DifferentialNeuronState; z++){
			aux=fabs(AuxNeuronState_p1[z]-AuxNeuronState_p[z]);
			if(aux>aux2){
				aux2=aux;
			}
		}

		memcpy(AuxNeuronState_p , AuxNeuronState_p1 ,sizeof(float)* N_DifferentialNeuronState);

		epsilon=aux2;
		k++;
	}

	if(State[index]<1){
		State[index]++;
	}



	for (int j=0; j<N_DifferentialNeuronState; j++){
		D[index*N_DifferentialNeuronState + j]=(AuxNeuronState_p[j]-NeuronState[j])/elapsed_time;
	}

	memcpy(NeuronState, AuxNeuronState_p ,sizeof(float)* N_DifferentialNeuronState);

	model->EvaluateTimeDependentEcuation(NeuronState, elapsed_time);

	return;

}

ostream & BDF1vs::PrintInfo(ostream & out){
	out << "Integration Method Type: " << this->GetType() << endl;

	return out;
}	

void BDF1vs::InitializeStates(int N_neurons, float * initialization){

	D = new float [N_neurons*N_DifferentialNeuronState];
	State = new int [N_neurons]();

	OriginalD = new float [N_neurons*N_DifferentialNeuronState];
	OriginalState = new int [N_neurons]();
}


void BDF1vs::resetState(int index){
	State[index]=0;
}

void BDF1vs::ReturnToOriginalState(int index){
	memcpy(D + (index*N_DifferentialNeuronState), OriginalD + (index*N_DifferentialNeuronState), N_DifferentialNeuronState*sizeof(float));
	State[index]=OriginalState[index];
}

