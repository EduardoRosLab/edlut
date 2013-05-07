/***************************************************************************
 *                           LIFTimeDrivenModelRK_GPU.h                    *
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

#ifndef LIFTIMEDRIVENMODELRK_GPU_H_
#define LIFTIMEDRIVENMODELRK_GPU_H_

/*!
 * \file LIFTimeDrivenModelRK_GPU.h
 *
 * \author Francisco Naveros
 * \date January 2012
 *
 * This file declares a class which abstracts a Leaky Integrate-And-Fire neuron model
 * using 4th order Runge-Kutta integration method.
 */

#include "./LIFTimeDrivenModel_GPU.h"

#include <string>

using namespace std;

class InputSpike;
class VectorNeuronState;
class Interconnection;

/*!
 * \class LIFTimeDrivenModelRK_GPU
 *
 * \brief Leaky Integrate-And-Fire Time-Driven neuron model using 4th order Runge-Kutta integration method.
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
class LIFTimeDrivenModelRK_GPU : public LIFTimeDrivenModel_GPU {
	public:

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 * \param NeuronTypeID Neuron type identificator
		 * \param NeuronModelID Neuron model identificator.
		 */
		LIFTimeDrivenModelRK_GPU(string NeuronTypeID, string NeuronModelID);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~LIFTimeDrivenModelRK_GPU();

		/*!
		 * \brief Update the neuron state variables.
		 *
		 * It updates the neuron state variables.
		 *
		 * \param index The cell index inside the VectorNeuronState. if index=-1, updating all cell.
		 * \param The current neuron state.
		 * \param CurrentTime Current time.
		 *
		 * \return True if an output spike have been fired. False in other case.
		 */
		virtual bool UpdateState(int index, VectorNeuronState * State, double CurrentTime);
};

#endif /* LIFTIMEDRIVENMODELRK_GPU_H_ */

