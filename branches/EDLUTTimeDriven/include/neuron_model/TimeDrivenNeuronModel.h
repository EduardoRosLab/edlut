/***************************************************************************
 *                           TimeDrivenNeuronModel.h                       *
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

#ifndef TIMEDRIVENNEURONMODEL_H_
#define TIMEDRIVENNEURONMODEL_H_

/*!
 * \file TimeDrivenNeuronModel.h
 *
 * \author Jesus Garrido
 * \date January 2011
 *
 * This file declares a class which abstracts an event-driven neuron model.
 */

#include "./NeuronModel.h"

#include <string>

using namespace std;

class InputSpike;
class NeuronState;

/*!
 * \class TimeDrivenNeuronModel
 *
 * \brief Time-Driven Spiking neuron model
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
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 * \param NeuronModelID Neuron model identificator.
		 */
		TimeDrivenNeuronModel(string NeuronModelID);

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
		 * \param The current neuron state.
		 * \param CurrentTime Current time.
		 *
		 * \return True if an output spike have been fired. False in other case.
		 */
		virtual bool UpdateState(NeuronState * State, double CurrentTime) = 0;

		/*!
		 * \brief It gets the neuron model type (event-driven or time-driven).
		 *
		 * It gets the neuron model type (event-driven or time-driven).
		 *
		 * \return The type of the neuron model.
		 */
		enum NeuronModelType GetModelType();

};

#endif /* NEURONMODEL_H_ */
