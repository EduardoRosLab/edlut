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

	public:

		/*!
		 * \brief Default constructor without parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 */
		NeuronModel();

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~NeuronModel();

		/*!
		 * \brief It initializes the neuron state to defined values.
		 *
		 * It initializes the neuron state to defined values.
		 *
		 * \param State Cell current state.
		 */
		virtual void InitializeState(NeuronState & State) = 0;

		/*!
		 * \brief It updates the neuron state after the evolution of the time.
		 *
		 * It updates the neuron state after the evolution of the time.
		 *
		 * \param State Cell current state.
		 * \param ElapsedTime Time elapsed from the previous update.
		 */
		virtual void UpdateState(NeuronState & State, double ElapsedTime) = 0;

		/*!
		 * \brief It abstracts the effect of an input spike in the cell.
		 *
		 * It abstracts the effect of an input spike in the cell.
		 *
		 * \param State Cell current state.
		 * \param InputConnection Input connection from which the input spike has got the cell.
		 */
		virtual void SynapsisEffect(NeuronState & State, const Interconnection * InputConnection) = 0;

		/*!
		 * \brief It returns the next spike time.
		 *
		 * It returns the next spike time.
		 *
		 * \param State Cell current state.
		 * \return The next firing spike time. -1 if no spike is predicted.
		 */
		virtual double NextFiringPrediction(NeuronState & State) = 0;

};

#endif /* NEURONMODEL_H_ */
