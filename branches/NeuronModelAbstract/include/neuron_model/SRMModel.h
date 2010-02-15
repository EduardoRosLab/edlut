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

#include "../spike/EDLUTFileException.h"

class SRMState;


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

	private:
		/*!
		 * \brief Decay time constant of the EPSP
		 */
		float tau;

		/*!
		 * \brief Resting potential
		 */
		float vr;

		/*!
		 * \brief Synaptic efficacy
		 */
		float W;

		/*!
		 * \brief Spontaneous firing rate
		 */
		float r0;

		/*!
		 * \brief Probabilistic threshold potential
		 */
		float v0;

		/*!
		 * \brief Gain factor
		 */
		float vf;

		/*!
		 * \brief Absolute refractory period
		 */
		float tauabs;

		/*!
		 * \brief Relative refractory period
		 */
		float taurel;

		/*!
		 * \brief Time step in simulation
		 */
		float timestep;

		/*!
		 * \brief Initial state of this neuron model
		 */
		SRMState * InitialState;


	protected:
		/*!
		 * \brief It calculates the potential difference between resting and the potential in the defined time.
		 *
		 * It calculates the potential difference between resting and the potential in the defined time.
		 *
		 * \param State Cell current state.
		 * \param CurrentTime Current simulation time.
		 *
		 * \return The potential difference between resting and the potential in the defined time.
		 */
		double PotentialIncrement(SRMState & State, double CurrentTime);

		/*!
		 * \brief It checks if an spike is fired in the defined time.
		 *
		 * It checks if an spike is fired in the defined time.
		 *
		 * \param State Cell current state.
		 * \param CurrentTime Current simulation time.
		 *
		 * \return True if an spike is fired in the defined time. False in otherwise.
		 */
		bool CheckSpikeAt(SRMState & State, double CurrentTime);

		/*!
		 * \brief It loads the neuron model description.
		 *
		 * It loads the neuron type description from the file .cfg.
		 *
		 * \param ConfigFile Name of the neuron description file (*.cfg).
		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		void LoadNeuronModel(string ConfigFile) throw (EDLUTFileException);

		/*!
		 * \brief It updates the neuron state after the evolution of the time.
		 *
		 * It updates the neuron state after the evolution of the time.
		 *
		 * \param State Cell current state.
		 * \param CurrentTime Current simulation time.
		 */
		virtual void UpdateState(SRMState & State, double CurrentTime);

		/*!
		 * \brief It abstracts the effect of an input spike in the cell.
		 *
		 * It abstracts the effect of an input spike in the cell.
		 *
		 * \param State Cell current state.
		 * \param InputConnection Input connection from which the input spike has got the cell.
		 */
		virtual void SynapsisEffect(SRMState & State, Interconnection * InputConnection);

		/*!
		 * \brief It returns the next spike time.
		 *
		 * It returns the next spike time.
		 *
		 * \param State Cell current state.
		 * \return The next firing spike time. -1 if no spike is predicted.
		 */
		virtual double NextFiringPrediction(SRMState & State);

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
		virtual NeuronState * InitializeState();

		/*!
		 * \brief It generates the first spike (if any) in a cell.
		 *
		 * It generates the first spike (if any) in a cell.
		 *
		 * \param Cell The cell to check if activity is generated.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * GenerateInitialActivity(Neuron &  Cell);

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
		virtual InternalSpike * ProcessInputSpike(PropagatedSpike &  InputSpike);

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
		virtual InternalSpike * GenerateNextSpike(const InternalSpike &  OutputSpike);

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
		virtual bool DiscardSpike(InternalSpike &  OutputSpike);

		/*!
		 * \brief It prints information about the load type.
		 *
		 * It prints information about the load type.
		 *
		 */
		virtual void GetModelInfo();
};

#endif /* SRMMODEL_H_ */
