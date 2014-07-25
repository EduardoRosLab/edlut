/***************************************************************************
 *                           TimeDrivenNeuronModel.cpp                     *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido                        *
 * email                : jgarrido@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/neuron_model/TimeDrivenNeuronModel.h"
#include "../../include/neuron_model/NeuronModel.h"

#include "../../include/openmp/openmp.h"

TimeDrivenNeuronModel::TimeDrivenNeuronModel(string NeuronTypeID, string NeuronModelID): NeuronModel(NeuronTypeID, NeuronModelID){
	// TODO Auto-generated constructor stub

}

TimeDrivenNeuronModel::~TimeDrivenNeuronModel() {
	delete integrationMethod;

	delete [] LimitOfOpenMPTasks;
}


enum NeuronModelType TimeDrivenNeuronModel::GetModelType(){
	return TIME_DRIVEN_MODEL_CPU;
}

void TimeDrivenNeuronModel::CalculateTaskSizes(int N_neurons, int minimumSize){
	//Calculate number of OpenMP task and size of each one.

	if((N_neurons/NumberOfOpenMPThreads)>=minimumSize){
		NumberOfOpenMPTasks=NumberOfOpenMPThreads;
	}else{
		if(N_neurons<=minimumSize){
			NumberOfOpenMPTasks=1;
		}else{
			for(int i=NumberOfOpenMPThreads-1; i>1; i--){
				if((N_neurons/i)>minimumSize){
					NumberOfOpenMPTasks=i;
				}
			}
		}
	}

	LimitOfOpenMPTasks = new int[NumberOfOpenMPTasks+1];
	LimitOfOpenMPTasks[0]=0;
	int aux=(N_neurons+NumberOfOpenMPTasks-1)/NumberOfOpenMPTasks;
	for(int i=1; i<NumberOfOpenMPTasks; i++){
		LimitOfOpenMPTasks[i]=aux*i;
	}
	LimitOfOpenMPTasks[NumberOfOpenMPTasks]=N_neurons;
}







