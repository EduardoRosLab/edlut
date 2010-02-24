/***************************************************************************
 *                           NeuronModel.h                                 *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
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

#ifndef NEURONMODEL_H_
#define NEURONMODEL_H_

/*!
 * \file NeuronModel.h
 *
 * \author Jesus Garrido
 * \date February 2010
 *
 * This file declares a class which abstracts a spiking neuron behavior.
 */

#include "../spike/Interconnection.h"
#include "../spike/InternalSpike.h"
#include "../spike/PropagatedSpike.h"
#include "../spike/Neuron.h"
#include "./NeuronState.h"

#include "../spike/EDLUTFileException.h"

#include <string>

using namespace std;


/*!
 * \class NeuronModel
 *
 * \brief Spiking neuron model
 *
 * This class abstracts the behavior of a neuron in a spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Jesus Garrido
 * \date February 2010
 */
class NeuronModel {
	private:

		/*!
		 * \brief Neuron model ID.
		 */
		string ModelID;

	protected:
		/*!
		 * \brief Initial state of this neuron model
		 */
		NeuronState * InitialState;

	public:

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 * \param NeuronModelID Neuron model identificator.
		 */
		NeuronModel(string NeuronModelID);

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
		virtual NeuronState * InitializeState() = 0;

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
		 * \brief It gets the neuron model ID.
		 *
		 * It gets the neuron model ID.
		 *
		 * \return The identificator of the neuron model.
		 */
		string GetModelID();

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

};

#endif /* NEURONMODEL_H_ */
