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


RK2::RK2(int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, int N_CPU_thread):FixedStep("RK2",N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState, N_CPU_thread, false, false){
	AuxNeuronState = new float [N_NeuronStateVariables*N_CPU_thread];
	AuxNeuronState1 = new float [N_NeuronStateVariables*N_CPU_thread];
	AuxNeuronState2 = new float [N_NeuronStateVariables*N_CPU_thread];
}

RK2::~RK2(){
	free (AuxNeuronState);
	free (AuxNeuronState1);
	free (AuxNeuronState2);
}
		
void RK2::NextDifferentialEcuationValue(int index, TimeDrivenNeuronModel * Model, float * NeuronState, float elapsed_time, int CPU_thread_index){
	float * offset_AuxNeuronState=AuxNeuronState+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_AuxNeuronState1=AuxNeuronState1+(N_NeuronStateVariables*CPU_thread_index);
	float * offset_AuxNeuronState2=AuxNeuronState2+(N_NeuronStateVariables*CPU_thread_index);

	//1st term
	Model->EvaluateDifferentialEcuation(NeuronState, offset_AuxNeuronState1);
	
	//2nd term
	memcpy(offset_AuxNeuronState, NeuronState,sizeof(float)*N_NeuronStateVariables);
	for (int j=0; j<N_DifferentialNeuronState; j++){
		offset_AuxNeuronState[j]= NeuronState[j] + offset_AuxNeuronState1[j]*elapsed_time;
	}


	Model->EvaluateTimeDependentEcuation(offset_AuxNeuronState, elapsed_time);
	Model->EvaluateDifferentialEcuation(offset_AuxNeuronState, offset_AuxNeuronState2);


	for (int j=0; j<N_DifferentialNeuronState; j++){
		NeuronState[j]+=(offset_AuxNeuronState1[j]+offset_AuxNeuronState2[j])*elapsed_time*0.5f;
	}

	Model->EvaluateTimeDependentEcuation(NeuronState, elapsed_time);
}

ostream & RK2::PrintInfo(ostream & out){
	out << "Integration Method Type: " << this->GetType() << endl;

	return out;
}	
