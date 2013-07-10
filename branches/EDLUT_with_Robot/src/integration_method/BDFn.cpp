/***************************************************************************
 *                           BDFn.cpp                                      *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
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

#include "../../include/integration_method/BDFn.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"

#include <math.h>

const float BDFn::Coeficient [7][7]={{1.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
{1.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f},
{2.0f/3.0f,4.0f/3.0f,-1.0f/3.0f,0.0f,0.0f,0.0f,0.0f},
{6.0f/11.0f,18.0f/11.0f,-9.0f/11.0f,2.0f/11.0f,0.0f,0.0f,0.0f},
{12.0f/25.0f,48.0f/25.0f,-36.0f/25.0f,16.0f/25.0f,-3.0f/25.0f,0.0f,0.0f},
{60.0f/137.0f,300.0f/137.0f,-300.0f/137.0f,200.0f/137.0f,-75.0f/137.0f,12.0f/137.0f,0.0f},
{60.0f/147.0f,360.0f/147.0f,-450.0f/147.0f,400.0f/147.0f,-225.0f/147.0f,72.0f/147.0f,-10.0f/147.0f}};


BDFn::BDFn(int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, int N_CPU_thread, int BDForder):FixedStep("BDFn", N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState, N_CPU_thread, true, true),BDForder(BDForder){
	AuxNeuronState = new float [N_NeuronStateVariables*N_CPU_thread];
	AuxNeuronState_p = new float [N_NeuronStateVariables*N_CPU_thread];
	AuxNeuronState_p1 = new float [N_NeuronStateVariables*N_CPU_thread];
	AuxNeuronState_c = new float [N_NeuronStateVariables*N_CPU_thread];
	jacnum = new float [N_DifferentialNeuronState*N_DifferentialNeuronState*N_CPU_thread];
	J = new float [N_DifferentialNeuronState*N_DifferentialNeuronState*N_CPU_thread];
	inv_J = new float [N_DifferentialNeuronState*N_DifferentialNeuronState*N_CPU_thread];
}

BDFn::~BDFn(){
	if(BDForder>1){
		for(int i=0; i<BDForder-1; i++){
			delete [] PreviousNeuronState[i];
		}
		delete [] PreviousNeuronState;
	}
	
	for(int i=0; i<BDForder; i++){
		delete [] D[i];
	}
	delete [] D;
	delete [] state;

	delete [] AuxNeuronState;
	delete [] AuxNeuronState_p;
	delete [] AuxNeuronState_p1;
	delete [] AuxNeuronState_c;
	delete [] jacnum;
	delete [] J;
	delete [] inv_J;

}
		
void BDFn::NextDifferentialEcuationValue(int index, TimeDrivenNeuronModel * Model, float * NeuronState, float elapsed_time, int CPU_thread_index){

	float * offset_AuxNeuronState = AuxNeuronState+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_AuxNeuronState_p = AuxNeuronState_p+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_AuxNeuronState_p1 = AuxNeuronState_p1+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_AuxNeuronState_c = AuxNeuronState_c+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_jacnum = jacnum+(N_DifferentialNeuronState*N_DifferentialNeuronState*CPU_thread_index);
	float * offset_J = J+(N_DifferentialNeuronState*N_DifferentialNeuronState*CPU_thread_index);
	float * offset_inv_J = inv_J+(N_DifferentialNeuronState*N_DifferentialNeuronState*CPU_thread_index);

	//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
	if(state[index]==0){
		Model->EvaluateDifferentialEcuation(NeuronState, offset_AuxNeuronState);
		for (int j=0; j<N_DifferentialNeuronState; j++){
			offset_AuxNeuronState_p[j]= NeuronState[j] + elapsed_time*offset_AuxNeuronState[j];
		}
	}
	//In this case we use the value of previous states to calculate an aproximation of the solution.
	else{
		for (int j=0; j<N_DifferentialNeuronState; j++){
			offset_AuxNeuronState_p[j]= NeuronState[j];
			for (int i=0; i<state[index]; i++){
				offset_AuxNeuronState_p[j]+=D[i][index*N_DifferentialNeuronState+j];
			}
		}
	}

	for(int i=N_DifferentialNeuronState; i<N_NeuronStateVariables; i++){
		offset_AuxNeuronState_p[i]=NeuronState[i];
	}


	Model->EvaluateTimeDependentEcuation(offset_AuxNeuronState_p,elapsed_time);



	float epsi=1.0f;
	int k=0;

	//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
	//epsi is the difference between two consecutive aproximation of the implicit method. 
	while (epsi>1e-16 && k<5){
		Model->EvaluateDifferentialEcuation(offset_AuxNeuronState_p, offset_AuxNeuronState);

		for (int j=0; j<N_DifferentialNeuronState; j++){
			offset_AuxNeuronState_c[j]=Coeficient[state[index]][0]*elapsed_time*offset_AuxNeuronState[j] + Coeficient[state[index]][1]*NeuronState[j];
			for (int i=1; i<state[index]; i++){
				offset_AuxNeuronState_c[j]+=Coeficient[state[index]][i+1]*PreviousNeuronState[i-1][index*N_DifferentialNeuronState + j];
			}
		}

		//jacobian.
		Jacobian(Model, offset_AuxNeuronState_p, offset_jacnum, CPU_thread_index, elapsed_time);
	
		for(int z=0; z<N_DifferentialNeuronState; z++){
			for(int t=0; t<N_DifferentialNeuronState; t++){
				offset_J[z*N_DifferentialNeuronState + t] = Coeficient[state[index]][0] * elapsed_time * offset_jacnum[z*N_DifferentialNeuronState + t];
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

		//We calculate the difference between both aproximations.
		float aux=0.0f;
		float aux2=0.0f;
		for(int z=0; z<N_DifferentialNeuronState; z++){
			aux=fabs(offset_AuxNeuronState_p1[z]-offset_AuxNeuronState_p[z]);
			if(aux>aux2){
				aux2=aux;
			}
		}

		memcpy(offset_AuxNeuronState_p , offset_AuxNeuronState_p1 ,sizeof(float)* N_DifferentialNeuronState);

		epsi=aux2;
		k++;
	}

	//We increase the state of the integration method.
	if(state[index]<BDForder){
		state[index]++;
	}



	//We acumulate these new values for the next step.
	for (int j=0; j<N_DifferentialNeuronState; j++){

		for(int i=(state[index]-1); i>0; i--){ 
			D[i][index*N_DifferentialNeuronState + j]=-D[i-1][index*N_DifferentialNeuronState + j];
		}
		D[0][index*N_DifferentialNeuronState + j]=offset_AuxNeuronState_p[j]-NeuronState[j];
		for(int i=1; i<state[index]; i++){ 
			D[i][index*N_DifferentialNeuronState + j]+=D[i-1][index*N_DifferentialNeuronState + j];
		}
	}

	if(state[index]>1){
		for(int i=state[index]-2; i>0; i--){
			memcpy(PreviousNeuronState[i] + (index*N_DifferentialNeuronState), PreviousNeuronState[i-1] + (index*N_DifferentialNeuronState) ,sizeof(float)* N_DifferentialNeuronState);
		}
		
		memcpy(PreviousNeuronState[0] + (index*N_DifferentialNeuronState), NeuronState ,sizeof(float)* N_DifferentialNeuronState);
	}
	memcpy(NeuronState, offset_AuxNeuronState_p ,sizeof(float)* N_DifferentialNeuronState);



	//Finaly, we evaluate the neural state variables with time dependence.
	Model->EvaluateTimeDependentEcuation(NeuronState, elapsed_time);

	return;

}

ostream & BDFn::PrintInfo(ostream & out){
	out << "Integration Method Type: " << this->GetType() << endl;

	return out;
}	

void BDFn::InitializeStates(int N_neurons, float * initialization){
	if(BDForder>1){
		PreviousNeuronState = (float **)new float* [BDForder-1];
		for(int i=0; i<(BDForder-1); i++){
			PreviousNeuronState[i] = new float [N_neurons*N_DifferentialNeuronState];
		}
	}
	D = (float **)new float* [BDForder];
	for(int i=0; i<BDForder; i++){
		D[i] = new float [N_neurons*N_DifferentialNeuronState];
	}


	state = new int [N_neurons]();
}


void BDFn::resetState(int index){
	state[index]=0;
}

