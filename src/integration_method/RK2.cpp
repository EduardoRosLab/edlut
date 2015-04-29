/***************************************************************************
 *                           RK2.cpp                                       *
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

#include "../../include/integration_method/RK2.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"


RK2::RK2(TimeDrivenNeuronModel * NewModel, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState):FixedStep(NewModel,"RK2",N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState, false, false){
}

RK2::~RK2(){
}
		
void RK2::NextDifferentialEcuationValue(int index, float * NeuronState, float elapsed_time){

	float AuxNeuronState[MAX_VARIABLES];
	float AuxNeuronState1[MAX_VARIABLES];
	float AuxNeuronState2[MAX_VARIABLES];


	//1st term
	this->model->EvaluateDifferentialEcuation(NeuronState, AuxNeuronState1, index);
	
	//2nd term
	memcpy(AuxNeuronState, NeuronState,sizeof(float)*N_NeuronStateVariables);
	for (int j=0; j<N_DifferentialNeuronState; j++){
		AuxNeuronState[j]= NeuronState[j] + AuxNeuronState1[j]*elapsed_time;
	}


	this->model->EvaluateTimeDependentEcuation(AuxNeuronState, elapsed_time);
	this->model->EvaluateDifferentialEcuation(AuxNeuronState, AuxNeuronState2, index);


	for (int j=0; j<N_DifferentialNeuronState; j++){
		NeuronState[j]+=(AuxNeuronState1[j]+AuxNeuronState2[j])*elapsed_time*0.5f;
	}

	this->model->EvaluateTimeDependentEcuation(NeuronState, elapsed_time);
}

ostream & RK2::PrintInfo(ostream & out){
	out << "Integration Method Type: " << this->GetType() << endl;

	return out;
}	
