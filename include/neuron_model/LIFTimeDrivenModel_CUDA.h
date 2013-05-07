/***************************************************************************
 *                           LIFTimeDrivenModel_GPU.h                      *
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

#ifndef LIFTIMEDRIVENMODEL_CUDA_H_
#define LIFTIMEDRIVENMODEL_CUDA_H_

/*!
 * \file LIFTimeDrivenModel_GPU.h
 *
 * \author Francisco Naveros
 * \date January 2012
 *
 * This file declares a class which abstracts a Leaky Integrate-And-Fire neuron model.
 */

#include "../../include/neuron_model/LIFTimeDrivenModel_GPU.h"

		//Library for CUDA
		#include <cutil_inline.h>

//#include <string>
//
//using namespace std;
//
//class InputSpike;
//class NeuronState;
//class Interconnection;

/*!
 * \class LIFTimeDrivenModel_GPU
 *
 * \brief Leaky Integrate-And-Fire Time-Driven neuron model
 *
 * This class abstracts the behavior of a neuron in a time-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date January 2012
 */




void createSynchronize();

void synchronizeGPU_CPU();

void destroySynchronize();


void UpdateStateGPU(float * parameter, float * AuxStateGPU, float * AuxStateCPU, float * VectorNeuronStates_GPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, bool * InternalSpikeCPU, int SizeStates, double CurrentTime);

void UpdateStateGPU(float * elapsed_time, float * parameter, float * AuxStateGPU, float * AuxStateCPU, float * VectorNeuronStates_GPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, bool * InternalSpikeCPU, int SizeStates, double CurrentTime);

void UpdateStateRKGPU(float * parameter, float * AuxStateGPU, float * AuxStateCPU, float * VectorNeuronStates_GPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, bool * InternalSpikeCPU, int SizeStates, double CurrentTime);

void UpdateStateRKGPU(float * elapsed_time, float * parameter, float * AuxStateGPU, float * AuxStateCPU, float * VectorNeuronStates_GPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, bool * InternalSpikeCPU, int SizeStates, double CurrentTime);

void InformationGPU();


#endif /* NEURONMODEL_H_ */
