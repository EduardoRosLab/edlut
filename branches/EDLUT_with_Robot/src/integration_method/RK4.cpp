/***************************************************************************
 *                           RK4.cpp                                       *
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

#include "../../include/integration_method/RK4.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"


RK4::RK4(TimeDrivenNeuronModel * NewModel, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState):FixedStep(NewModel,"RK4", N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState, false, false){

}

RK4::~RK4(){
}
		

void RK4::NextDifferentialEcuationValue(int index, float * NeuronState, float elapsed_time){
	int j;

	float AuxNeuronState[MAX_VARIABLES];
	float AuxNeuronState1[MAX_VARIABLES];
	float AuxNeuronState2[MAX_VARIABLES];
	float AuxNeuronState3[MAX_VARIABLES];
	float AuxNeuronState4[MAX_VARIABLES];

	const float elapsed_time_0_5=elapsed_time*0.5f;
	const float elapsed_time_0_16=elapsed_time*0.166666666667f;


	//1st term
	this->model->EvaluateDifferentialEcuation(NeuronState, AuxNeuronState1, index);
	
	//2nd term
	for (j=0; j<N_DifferentialNeuronState; j++){
		AuxNeuronState[j]= NeuronState[j] + AuxNeuronState1[j]*elapsed_time_0_5;
	}
	for (j=N_DifferentialNeuronState; j<N_NeuronStateVariables; j++){
		AuxNeuronState[j]= NeuronState[j];
	}

	this->model->EvaluateTimeDependentEcuation(AuxNeuronState, elapsed_time_0_5);
	this->model->EvaluateDifferentialEcuation(AuxNeuronState, AuxNeuronState2, index);

	//3rd term
	for (j=0; j<N_DifferentialNeuronState; j++){
		AuxNeuronState[j]=NeuronState[j] + AuxNeuronState2[j]*elapsed_time_0_5;
	}

	this->model->EvaluateDifferentialEcuation(AuxNeuronState, AuxNeuronState3, index);

	//4rd term
	for (j=0; j<N_DifferentialNeuronState; j++){
		AuxNeuronState[j]=NeuronState[j] + AuxNeuronState3[j]*elapsed_time;
	}

	this->model->EvaluateTimeDependentEcuation(AuxNeuronState, elapsed_time_0_5);
	this->model->EvaluateDifferentialEcuation(AuxNeuronState, AuxNeuronState4, index);


	for (j=0; j<N_DifferentialNeuronState; j++){
		NeuronState[j]+=(AuxNeuronState1[j]+2.0f*(AuxNeuronState2[j]+AuxNeuronState3[j])+AuxNeuronState4[j])*elapsed_time_0_16;
	}

	this->model->EvaluateTimeDependentEcuation(NeuronState, elapsed_time);

}


ostream & RK4::PrintInfo(ostream & out){
	out << "Integration Method Type: " << this->GetType() << endl;

	return out;
}	
