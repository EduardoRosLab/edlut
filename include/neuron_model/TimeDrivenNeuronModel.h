/***************************************************************************
 *                           TimeDrivenNeuronModel.h                       *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Jesus Garrido and Francisco Naveros  *
 * email                : jgarrido@atc.ugr.es, fnaveros@atc.ugr.es         *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TIMEDRIVENNEURONMODEL_H_
#define TIMEDRIVENNEURONMODEL_H_

/*!
 * \file TimeDrivenNeuronModel.h
 *
 * \author Jesus Garrido
 * \author Francisco Naveros
 * \date January 2011
 *
 * This file declares a class which abstracts an time-driven neuron model in a CPU.
 */

#include "./NeuronModel.h"

#include "../simulation/LoadTimeEvent.h"

#include "../integration_method/IntegrationMethod.h"
#include "../integration_method/LoadIntegrationMethod.h"

#include <string>

using namespace std;

class InputSpike;
class VectorNeuronState;



/*!
 * \class TimeDrivenNeuronModel
 *
 * \brief Time-Driven Spiking neuron model in a CPU
 *
 * This class abstracts the behavior of a neuron in a time-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Jesus Garrido
 * \date January 2011
 */
class TimeDrivenNeuronModel : public NeuronModel {
	public:

		/*!
		 * \brief integration method.
		*/
		IntegrationMethod * integrationMethod;


		/*!
		 * \brief number of OpenMP task in which this neuron model will be divided. This variable is 
		 *  calculated in CalculateTaskSizes.
		*/
		int NumberOfOpenMPTasks;

		/*!
		 * \brief block limits of the OpenMP task. This variable is calculated in CalculateTaskSizes.
		*/
		int * LimitOfOpenMPTasks;


		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 * \param NeuronTypeID Neuron model type.
		 * \param NeuronModelID Neuron model description file.
		 */
		TimeDrivenNeuronModel(string NeuronTypeID, string NeuronModelID);


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~TimeDrivenNeuronModel();

		
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
		virtual void InitializeStates(int N_neurons)=0;


		/*!
		 * \brief It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
		 *
		 * It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
		 *
		 * \param NeuronState value of the neuron state variables where differential equations are evaluated.
		 * \param AuxNeuronState results of the differential equations evaluation.
		 */
		virtual void EvaluateDifferentialEcuation(float * NeuronState, float * AuxNeuronState)=0;


		/*!
		 * \brief It evaluates the time depedendent ecuation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * It evaluates the time depedendent ecuation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * \param NeuronState value of the neuron state variables where time dependent equations are evaluated.
		 * \param elapsed_time integration time step.
		 */
		virtual void EvaluateTimeDependentEcuation(float * NeuronState, float elapsed_time)=0;

		/*!
		 * \brief It calculate for the neuron model of each OpenMP queue the number of OpenMP 
		 *  tasks and the size of each one.
		 *
		 *  It calculate for the neuron model of each OpenMP queue the number of OpenMP 
		 *  tasks and the size of each one.		 
		 *
		 * \param N_neurons number of neuron in this queue for this neuron model.
		 * \param minimumSize minimum number of neurons that must contain a task.
		 */
		void CalculateTaskSizes(int N_neurons, int minimumSize);


};

#endif /* TIMEDRIVENNEURONMODEL_H_ */
