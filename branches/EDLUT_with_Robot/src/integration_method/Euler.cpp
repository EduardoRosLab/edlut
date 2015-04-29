/***************************************************************************
 *                           Euler.cpp                                     *
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

#include "../../include/integration_method/Euler.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"


Euler::Euler(TimeDrivenNeuronModel * NewModel, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState):FixedStep(NewModel,"Euler",N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState, false, false){
}

Euler::~Euler(){
}
		
void Euler::NextDifferentialEcuationValue(int index,float * NeuronState, float elapsed_time){
	float AuxNeuronState[MAX_VARIABLES];
	
	this->model->EvaluateDifferentialEcuation(NeuronState, AuxNeuronState, index);

	for (int j=0; j<N_DifferentialNeuronState; j++){
		NeuronState[j]+=elapsed_time*AuxNeuronState[j];
	}

	this->model->EvaluateTimeDependentEcuation(NeuronState, elapsed_time);
}

ostream & Euler::PrintInfo(ostream & out){
	out << "Integration Method Type: " << this->GetType() << endl;

	return out;
}	
