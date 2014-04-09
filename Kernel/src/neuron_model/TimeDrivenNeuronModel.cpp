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

#include <string>
#ifdef _OPENMP
	#include <omp.h>
#endif

TimeDrivenNeuronModel::TimeDrivenNeuronModel(string NeuronTypeID, string NeuronModelID): NeuronModel(NeuronTypeID, NeuronModelID){
	// TODO Auto-generated constructor stub
	
	#ifdef _OPENMP
		N_CPU_thread=omp_get_max_threads();
	#else
		N_CPU_thread=1;
	#endif
}

TimeDrivenNeuronModel::~TimeDrivenNeuronModel() {
	delete integrationMethod;
}


enum NeuronModelType TimeDrivenNeuronModel::GetModelType(){
	return TIME_DRIVEN_MODEL_CPU;
}








