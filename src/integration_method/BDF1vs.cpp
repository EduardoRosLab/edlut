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


BDF1vs::BDF1vs(int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, int N_CPU_thread):VariableStep("BDF1vs", N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState, N_CPU_thread, true, true){

	AuxNeuronState = new float [N_NeuronStateVariables*N_CPU_thread];
	AuxNeuronState_p = new float [N_NeuronStateVariables*N_CPU_thread];
	AuxNeuronState_p1 = new float [N_NeuronStateVariables*N_CPU_thread];
	AuxNeuronState_c = new float [N_NeuronStateVariables*N_CPU_thread];
	jacnum = new float [N_DifferentialNeuronState*N_DifferentialNeuronState*N_CPU_thread];
	J = new float [N_DifferentialNeuronState*N_DifferentialNeuronState*N_CPU_thread];
	inv_J = new float [N_DifferentialNeuronState*N_DifferentialNeuronState*N_CPU_thread];
}

BDF1vs::~BDF1vs(){
	delete [] D;
	delete [] OriginalD;
	delete [] State;
	delete [] OriginalState;

	delete [] AuxNeuronState;
	delete [] AuxNeuronState_p;
	delete [] AuxNeuronState_p1;
	delete [] AuxNeuronState_c;
	delete [] jacnum;
	delete [] J;
	delete [] inv_J;

}
		
void BDF1vs::NextDifferentialEcuationValue(int index, TimeDrivenNeuronModel * Model, float * NeuronState, float elapsed_time, int CPU_thread_index){

	float * offset_AuxNeuronState = AuxNeuronState+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_AuxNeuronState_p = AuxNeuronState_p+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_AuxNeuronState_p1 = AuxNeuronState_p1+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_AuxNeuronState_c = AuxNeuronState_c+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_jacnum = jacnum+(N_DifferentialNeuronState*N_DifferentialNeuronState*CPU_thread_index);
	float * offset_J = J+(N_DifferentialNeuronState*N_DifferentialNeuronState*CPU_thread_index);
	float * offset_inv_J = inv_J+(N_DifferentialNeuronState*N_DifferentialNeuronState*CPU_thread_index);

	memcpy(OriginalD + (index*N_DifferentialNeuronState), D + (index*N_DifferentialNeuronState), N_DifferentialNeuronState*sizeof(float));
	OriginalState[index]=State[index];


	if(State[index]==0){
		Model->EvaluateDifferentialEcuation(NeuronState, offset_AuxNeuronState);
		for (int j=0; j<N_DifferentialNeuronState; j++){
			offset_AuxNeuronState_p[j]= NeuronState[j] + elapsed_time*offset_AuxNeuronState[j];
		}
	}else{
		for (int j=0; j<N_DifferentialNeuronState; j++){
			offset_AuxNeuronState_p[j]= NeuronState[j] + elapsed_time*D[index*N_DifferentialNeuronState+j];
		}
	}

	//memcpy(offset_AuxNeuronState_p + N_DifferentialNeuronState, NeuronState + N_DifferentialNeuronState ,sizeof(float)* (N_NeuronStateVariables-N_DifferentialNeuronState));
	for(int i=N_DifferentialNeuronState; i<N_NeuronStateVariables; i++){
		offset_AuxNeuronState_p[i]=NeuronState[i];
	}
		
	Model->EvaluateTimeDependentEcuation(offset_AuxNeuronState_p,elapsed_time);

	float epsilon=1.0;
	int k=0;

	while (epsilon>1e-16 && k<5){
		Model->EvaluateDifferentialEcuation(offset_AuxNeuronState_p, offset_AuxNeuronState);
		for (int j=0; j<N_DifferentialNeuronState; j++){
			offset_AuxNeuronState_c[j]=NeuronState[j] + elapsed_time*offset_AuxNeuronState[j];
		}

		//jacobian.
		Jacobian(Model, offset_AuxNeuronState_p, offset_jacnum, CPU_thread_index, elapsed_time);
	
		for(int z=0; z<N_DifferentialNeuronState; z++){
			for(int t=0; t<N_DifferentialNeuronState; t++){
				offset_J[z*N_DifferentialNeuronState + t] = elapsed_time * offset_jacnum[z*N_DifferentialNeuronState + t];
				if(z==t){
					offset_J[z*N_DifferentialNeuronState + t]-=1;
				}
			}
		}
		this->invermat(offset_J,offset_inv_J, CPU_thread_index);
		for(int z=0; z<N_DifferentialNeuronState; z++){
			float aux=0.0;
			for (int t=0; t<N_DifferentialNeuronState; t++){
				aux+=offset_inv_J[z*N_DifferentialNeuronState+t]*(offset_AuxNeuronState_p[t]-offset_AuxNeuronState_c[t]);
			}
			offset_AuxNeuronState_p1[z]=aux + offset_AuxNeuronState_p[z];
		}

		float aux=0.0;
		float aux2=0.0;
		for(int z=0; z<N_DifferentialNeuronState; z++){
			aux=fabs(offset_AuxNeuronState_p1[z]-offset_AuxNeuronState_p[z]);
			if(aux>aux2){
				aux2=aux;
			}
		}

		memcpy(offset_AuxNeuronState_p , offset_AuxNeuronState_p1 ,sizeof(float)* N_DifferentialNeuronState);

		epsilon=aux2;
		k++;
	}

	if(State[index]<1){
		State[index]++;
	}



	for (int j=0; j<N_DifferentialNeuronState; j++){
		D[index*N_DifferentialNeuronState + j]=(offset_AuxNeuronState_p[j]-NeuronState[j])/elapsed_time;
	}

	memcpy(NeuronState, offset_AuxNeuronState_p ,sizeof(float)* N_DifferentialNeuronState);

	Model->EvaluateTimeDependentEcuation(NeuronState, elapsed_time);

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

