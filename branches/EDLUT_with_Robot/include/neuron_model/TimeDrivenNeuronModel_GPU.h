/***************************************************************************
 *                           TimeDrivenNeuronModel_GPU.h                   *
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

#ifndef TIMEDRIVENNEURONMODEL_GPU_H_
#define TIMEDRIVENNEURONMODEL_GPU_H_

/*!
 * \file TimeDrivenNeuronModel_GPU.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which abstracts an time-driven neuron model in CPU for GPU.
 */

#include "./NeuronModel.h"

#include "../simulation/LoadTimeEvent_GPU.h"

#include "../integration_method/IntegrationMethod_GPU.h"
#include "../integration_method/LoadIntegrationMethod_GPU.h"

#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU2.h"

#include <string>




using namespace std;

class InputSpike;
class VectorNeuronState;





/*!
 * \class TimeDrivenNeuronModel_GPU
 *
 * \brief Time-Driven Spiking neuron model in CPU for GPU
 *
 * This class abstracts the behavior of a neuron in a time-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date November 2012
 */
class TimeDrivenNeuronModel_GPU : public NeuronModel {
	public:

		/*!
		 * \brief Number of CUDA threads.
		*/
		int N_thread;

		/*!
		 * \brief Number of CUDA blocks.
		*/
		int N_block;

		/*!
		 * \brief integration time step.
		*/
		double TimeDrivenStep_GPU;

		/*!
		 * \brief integration method in CPU for GPU.
		*/
		IntegrationMethod_GPU * integrationMethod_GPU;


		/*!
		 * \brief barrier to synchronize the CPU and the GPU.
		 */
		cudaEvent_t stop;

		/*!
		 * \brief GPU properties
		 */
		cudaDeviceProp prop;

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 * \param NeuronTypeID Neuron model type.
		 * \param NeuronModelID Neuron model description file.
		 */
		TimeDrivenNeuronModel_GPU(string NeuronTypeID, string NeuronModelID);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~TimeDrivenNeuronModel_GPU();

		
		/*!
		 * \brief Update the neuron state variables.
		 *
		 * It updates the neuron state variables.
		 *
		 * \param index The cell index inside the vector. if index=-1, updating all cell. 
		 * \param The current neuron state.
		 * \param CurrentTime Current time.
		 *
		 * \return True if an output spike have been fired. False in other case.
		 */
		virtual bool UpdateState(int index, VectorNeuronState * State, double CurrentTime) = 0;


		/*!
		 * \brief It gets the neuron model type (event-driven or time-driven).
		 *
		 * It gets the neuron model type (event-driven or time-driven).
		 *
		 * \return The type of the neuron model.
		 */
		enum NeuronModelType GetModelType();


		/*!
		 * \brief It initialice VectorNeuronState.
		 *
		 * It initialice VectorNeuronState.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 */
		virtual void InitializeStates(int N_neurons, int OpenMPQueueIndex)=0;


		/*!
		 * \brief It gets the time driven step.
		 *
		 * It gets the time driven step.
		 *
		 * \return The time driven step.
		 */
		double GetTimeDrivenStep_GPU();


		/*!
		 * \brief It initialice a neuron model in GPU.
		 *
		 * It initialice a neuron model in GPU.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 */
		virtual void InitializeClassGPU2(int N_neurons)=0;


		/*!
		 * \brief It delete a neuron model in GPU.
		 *
		 * It delete a neuron model in GPU.
		 */
		virtual void DeleteClassGPU2()=0;

		/*!
		 * \brief It create a object of type VectorNeuronState_GPU2 in GPU.
		 *
		 * It create a object of type VectorNeuronState_GPU2 in GPU.
		 */
		virtual void InitializeVectorNeuronState_GPU2()=0;


		/*!
		 * \brief It Checks if the neuron model has this connection type.
		 *
		 * It Checks if the neuron model has this connection type.
		 *
		 * \param Type input connection type.
		 *
		 * \return A a valid connection type for this neuron model.
		 */
		virtual int CheckSynapseTypeNumber(int Type)=0;
};

#endif /* TIMEDRIVENNEURONMODEL_GPU_H_ */
