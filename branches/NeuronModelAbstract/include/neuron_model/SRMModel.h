/***************************************************************************
 *                           SRMModel.h                                    *
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

#ifndef SRMMODEL_H_
#define SRMMODEL_H_

/*!
 * \file SRMModel.h
 *
 * \author Jesus Garrido
 * \date February 2010
 *
 * This file declares a class which implements a SRM (Spike response model) neuron
 * model.
 */

#include "./NeuronModel.h"
#include "./BufferedState.h"


/*!
 * \class SRMModel
 *
 * \brief Spike Response neuron model non-based in look-up tables.
 *
 * This class implements the behavior of a neuron in a spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This behavior is calculated based in a buffer of activity.
 *
 * \author Jesus Garrido
 * \date February 2010
 */
class SRMModel: public NeuronModel {

	public:
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object.
		 *
		 * \param NeuronModelID Neuron model identificator.
		 */
		SRMModel(string NeuronModelID);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~SRMModel();

		/*!
		 * \brief It initializes the neuron state to defined values.
		 *
		 * It initializes the neuron state to defined values.
		 *
		 * \param State Cell current state.
		 */
		virtual BufferedState * InitializeState();

		/*!
		 * \brief It updates the neuron state after the evolution of the time.
		 *
		 * It updates the neuron state after the evolution of the time.
		 *
		 * \param State Cell current state.
		 * \param ElapsedTime Time elapsed from the previous update.
		 */
		virtual void UpdateState(BufferedState & State, double ElapsedTime);

		/*!
		 * \brief It abstracts the effect of an input spike in the cell.
		 *
		 * It abstracts the effect of an input spike in the cell.
		 *TableBased
		 * \param State Cell current state.
		 * \param InputConnection Input connection from which the input spike has got the cell.
		 */
		virtual void SynapsisEffect(BufferedState & State, const Interconnection * InputConnection);

		/*!TableBased
		 * \brief It returns the next spike time.
		 *
		 * It returns the next spike time.
		 *
		 * \param State Cell current state.
		 * \return The next firing spike time. -1 if no spike is predicted.
		 */
		virtual double NextFiringPrediction(BufferedState & State);
};

#endif /* SRMMODEL_H_ */
