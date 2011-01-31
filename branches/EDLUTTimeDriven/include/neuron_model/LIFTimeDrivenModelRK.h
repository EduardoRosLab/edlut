/***************************************************************************
 *                           LIFTimeDrivenModelRK.h                        *
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

#ifndef LIFTIMEDRIVENMODELRK_H_
#define LIFTIMEDRIVENMODELRK_H_

/*!
 * \file LIFTimeDrivenModelRK.h
 *
 * \author Jesus Garrido
 * \date January 2011
 *
 * This file declares a class which abstracts a Leaky Integrate-And-Fire neuron model
 * using 4th order Runge-Kutta integration method.
 */

#include "./LIFTimeDrivenModel.h"

#include <string>

using namespace std;

class InputSpike;
class NeuronState;
class Interconnection;

/*!
 * \class LIFTimeDrivenModelRK
 *
 * \brief Leaky Integrate-And-Fire Time-Driven neuron model using 4th order Runge-Kutta integration method.
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
class LIFTimeDrivenModelRK : public LIFTimeDrivenModel {
	public:

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 * \param NeuronModelID Neuron model identificator.
		 */
		LIFTimeDrivenModelRK(string NeuronModelID);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~LIFTimeDrivenModelRK();

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
		virtual bool UpdateState(NeuronState * State, double CurrentTime);
};

#endif /* NEURONMODEL_H_ */

