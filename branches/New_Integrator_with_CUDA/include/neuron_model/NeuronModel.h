/***************************************************************************
 *                           NeuronModel.h                                 *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido and Francisco Naveros  *
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

#ifndef NEURONMODEL_H_
#define NEURONMODEL_H_

/*!
 * \file NeuronModel.h
 *
 * \author Jesus Garrido
 * \author Francisco Naveros
 * \date January 2011
 *
 * \note Modified on January 2012 in order to include time-driven simulation support in GPU.
 * New state variables (TIME_DRIVEN_MODEL_CPU and TIME_DRIVEN_MODEL_GPU)
 *
 * This file declares a class which abstracts an spiking neural model.
 */

#include <string>

#include "../spike/EDLUTFileException.h"

using namespace std;

class VectorNeuronState;
class InternalSpike;
class PropagatedSpike;

enum NeuronModelType {EVENT_DRIVEN_MODEL, TIME_DRIVEN_MODEL_CPU, TIME_DRIVEN_MODEL_GPU};


/*!
 * \class NeuronModel
 *
 * \brief Spiking neuron model
 *
 * This class abstracts the behavior of a neuron in a spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Jesus Garrido
 * \date January 2011
 */
class NeuronModel {
	private:

		/*!
		 * \brief Neuron type ID.
		*/
		string TypeID;

		/*!
		 * \brief Neuron model ID.
		 */
		string ModelID;

	protected:
		/*!
		 * \brief Initial state of this neuron model
		 */
		VectorNeuronState * InitialState;

	public:

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 * \param NeuronModelID Neuron model identificator.
		 */
		NeuronModel(string NeuronTypeID,string NeuronModelID);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~NeuronModel();

		/*!
		 * \brief It loads the neuron model description and tables (if necessary).
		 *
		 * It loads the neuron model description and tables (if necessary).
		 */
		virtual void LoadNeuronModel() throw (EDLUTFileException) = 0;

		/*!
		 * \brief It initializes the neuron state to defined values.
		 *
		 * It initializes the neuron state to defined values.
		 *
		 * \param State Cell current state.
		 */
		virtual VectorNeuronState * InitializeState() = 0;

		/*!
		 * \brief It processes a propagated spike (input spike in the cell).
		 *
		 * It processes a propagated spike (input spike in the cell).
		 *
		 * \note This function doesn't generate the next propagated spike. It must be externally done.
		 *
		 * \param InputSpike The spike happened.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * ProcessInputSpike(PropagatedSpike *  InputSpike) = 0;

		/*!
		 * \brief It gets the neuron type ID.
		 *
		 * It gets the neuron type ID.
		 *
		 * \return The identificator of the neuron type.
		 */
		string GetTypeID();

		/*!
		 * \brief It gets the neuron model ID.
		 *
		 * It gets the neuron model ID.
		 *
		 * \return The identificator of the neuron model.
		 */
		string GetModelID();

		/*!
		 * \brief It gets the neuron model type (event-driven or time-driven).
		 *
		 * It gets the neuron model type (event-driven or time-driven).
		 *
		 * \return The type of the neuron model.
		 */
		virtual enum NeuronModelType GetModelType() = 0;

		/*!
		 * \brief It prints the neuron model info.
		 *
		 * It prints the current neuron model characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out) = 0;

		/*!
		 * \brief It gets the VectorNeuronState.
		 *
		 * It gets the VectorNeuronState.
		 *
		 * \return The VectorNeuronState.
		 */
		VectorNeuronState * GetVectorNeuronState();

		/*!
		 * \brief It initialice VectorNeuronState.
		 *
		 * It initialice VectorNeuronState.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 */
		virtual void InitializeStates(int N_neurons)=0;




};

#endif /* NEURONMODEL_H_ */
