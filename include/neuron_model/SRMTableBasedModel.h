/***************************************************************************
 *                           SRMTableBasedModel.h                          *
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

#ifndef SRMTABLEBASEDMODEL_H_
#define SRMTABLEBASEDMODEL_H_

/*!
 * \file SRMTableBasedModel.h
 *
 * \author Jesus Garrido
 * \date February 2010
 *
 * This file declares a class which implements a neuron model based in
 * look-up tables.
 */

#include "TableBasedModel.h"

/*!
 * \class TableBasedModel
 *
 * \brief SRM (Spike-Response model) Spiking neuron model based in look-up tables
 *
 * This class implements the behavior of a SRM neuron. The firing time is calculated in
 * an stochastic way. It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This behavior is calculated based in precalculated look-up tables.
 *
 * \author Jesus Garrido
 * \date February 2010
 */
class SRMTableBasedModel: public TableBasedModel {
	protected:
		/*!
		 * \brief Number of the seed variable
		 */
		unsigned int SeedVar;

		/*!
		 * \brief Last spike time variable
		 */
		unsigned int LastSpikeVar;

		/*!
		 * \brief It loads the neuron model description.
		 *
		 * It loads the neuron type description from the file .cfg.
		 *
		 * \param ConfigFile Name of the neuron description file (*.cfg).
		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		virtual void LoadNeuronModel(string ConfigFile) throw (EDLUTFileException);

		/*!
		 * \brief It abstracts the effect of an input spike in the cell.
		 *
		 * It abstracts the effect of an input spike in the cell.
		 *
		 * \param State Cell current state.
		 * \param InputConnection Input connection from which the input spike has got the cell.
		 */
		virtual void SynapsisEffect(int index, Interconnection * InputConnection);

		/*!
		 * \brief It returns the end of the refractory period.
		 *
		 * It returns the end of the refractory period.
		 *
		 * \param State Cell current state.
		 * \return The end of the refractory period. -1 if no spike is predicted.
		 */
		virtual double EndRefractoryPeriod(int index, VectorNeuronState * State);

		/*!
		 * \brief It updates the neuron state after the evolution of the time.
		 *
		 * It updates the neuron state after the evolution of the time.
		 *
		 * \param State Cell current state.
		 * \param CurrentTime Current simulation time.
		 */
		virtual void UpdateState(int index, VectorNeuronState * State, double CurrentTime);

		/*!
		 * \brief It returns the next spike time.
		 *
		 * It returns the next spike time.
		 *
		 * \param State Cell current state.
		 * \return The next firing spike time. -1 if no spike is predicted.
		 */
		virtual double NextFiringPrediction(int index, VectorNeuronState * State);

	public:

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object loading the configuration of
		 * the model and the look-up tables.
		 *
		 * \param NeuronModelID Neuron model identificator.
		 */
		SRMTableBasedModel(string NeuronTypeID, string NeuronModelID);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~SRMTableBasedModel();

		/*!
		 * \brief It loads the neuron model description and tables (if necessary).
		 *
		 * It loads the neuron model description and tables (if necessary).
		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		virtual void LoadNeuronModel() throw (EDLUTFileException);

		/*!
		 * \brief It creates the neuron state and initializes to defined values.
		 *
		 * It creates the neuron state and initializes to defined values.
		 *
		 * \return A new object with the neuron state.
		 */
		virtual VectorNeuronState * InitializeState();

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
		virtual InternalSpike * GenerateNextSpike(InternalSpike *  OutputSpike);

virtual void InitializeStates(int N_neurons);
};

#endif /* SRMTABLEBASEDMODEL_H_ */
