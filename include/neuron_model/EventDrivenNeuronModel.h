/***************************************************************************
 *                           EventDrivenNeuronModel.h                      *
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

#ifndef EVENTDRIVENNEURONMODEL_H_
#define EVENTDRIVENNEURONMODEL_H_

/*!
 * \file EventDrivenNeuronModel.h
 *
 * \author Jesus Garrido
 * \date January 2011
 *
 * This file declares a class which abstracts an event-driven neuron model.
 */

#include "./NeuronModel.h"

#include "../spike/EDLUTFileException.h"

#include <iostream>

using namespace std;

class Neuron;


/*!
 * \class EventDrivenNeuronModel
 *
 * \brief Event-Driven Spiking neuron model
 *
 * This class abstracts the behavior of a neuron in an event-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Jesus Garrido
 * \date January 2011
 */
class EventDrivenNeuronModel : public NeuronModel {
	public:

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 * \param NeuronModelID Neuron model identificator.
		 */
		EventDrivenNeuronModel(string NeuronModelID);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~EventDrivenNeuronModel();

		/*!
		 * \brief It generates the first spike (if any) in a cell.
		 *
		 * It generates the first spike (if any) in a cell.
		 *
		 * \param Cell The cell to check if activity is generated.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * GenerateInitialActivity(Neuron *  Cell) = 0;

		/*!
		 * \brief It processes an internal spike (generated spike in the cell).
		 *
		 * It processes an internal spike (generated spike in the cell).
		 *
		 * \note This function doesn't generate the next propagated (output) spike. It must be externally done.
		 * \note Before generating next spike, you should check if this spike must be discard.
		 *
		 * \see DiscardSpike
		 *
		 * \param OutputSpike The spike happened.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * GenerateNextSpike(InternalSpike *  OutputSpike) = 0;

		/*!
		 * \brief Check if the spike must be discard.
		 *
		 * Check if the spike must be discard. A spike must be discard if there are discrepancies between
		 * the next predicted spike and the spike time.
		 *
		 * \param OutputSpike The spike happened.
		 *
		 * \return True if the spike must be discard. False in otherwise.
		 */
		virtual bool DiscardSpike(InternalSpike *  OutputSpike) = 0;

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
