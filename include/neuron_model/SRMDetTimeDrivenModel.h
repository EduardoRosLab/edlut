/***************************************************************************
 *                           SRMDetTimeDrivenModel.h                       *
 *                           ----------------------                        *
 * copyright            : (C) 2013 by Jesus Garrido and Francisco Naveros  *
 * email                : jesusgarrido@ugr.es, fnaveros@ugr.es             *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef SRMDETTIMEDRIVENMODEL_H_
#define SRMDETTIMEDRIVENMODEL_H_

/*!
 * \file SRMDetTimeDrivenModel.h
 *
 * \author Jesus Garrido
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implements a SRM (Spike response model) time-driven neuron
 * model based on the model used by (Masquellier et al. 2011) "Spike Timing Dependent Plasticity Finds the Start of
 * Repeating Patterns in Continuous Spike Trains".
 *
 * \note: According to the last trend in EDLUT cell model development it follows the vectorial storing and processing
 * of the neuron state.
 */

#include "./TimeDrivenNeuronModel.h"

#include "../spike/EDLUTFileException.h"

class VectorSRMState;
class VectorBufferedState;
class Interconnection;

/*!
 * \class SRMDetTimeDrivenModel
 *
 * \brief Spike Response time-driven neuron model with deterministic firing mechanism..
 *
 * This class implements the behavior of a neuron in a spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect...). At this version, this model only
 * includes one input channel (excitatory). This behavior is calculated based in a buffer of activity.
 * More information about this cell model can be found at (Masquellier et al. 2011, 
 * "Spike Timing Dependent Plasticity Finds the Start of Repeating Patterns in Continuous Spike Trains", Plos One).
 *
 * \author Jesus Garrido
 * \author Francisco Naveros
 * \date May 2013
 */
class SRMDetTimeDrivenModel: public TimeDrivenNeuronModel {

	private:

		/*!
		 * \brief Membrane potential of the cell: tau_m
		 */
		float tau_m;

		/*!
		 * \brief Synaptic time constant: tau_s
		 */
		float tau_s;

		/*!
		 * \brief Spike threshold: T
		 */
		float threshold;

		/*!
		 * \brief Refractory period
		 */
		float refractory;

		/*!
		 * \brief Normalization factor: K
		 */
		float k;

		/*!
		 * \brief Positive pulse amplitude (K1).
		 */
		float k1;

		/*!
		 * \brief Negative spike-afterpotiential amplitude (K2).
		 */
		float k2;

		/*!
		 * \brief Time window in which the incoming spikes stay in the activity buffer (in the paper 7*tau_m).
		 */
		float buffer_amplitude;

	protected:
		/*!
		 * \brief It calculates the potential difference between resting and the potential in the defined time.
		 *
		 * It calculates the potential difference between resting and the potential in the defined time.
		 *
		 * \param index The cell index inside the VectorSRMState.
		 * \param State Cells state vector.
		 *
		 * \return The potential difference between resting and the potential in the defined time.
		 */
		float PotentialIncrement(int index, VectorSRMState * State);

		/*!
		 * \brief It checks if an spike is fired in the defined time.
		 *
		 * It checks if an spike is fired in the defined time.
		 *
		 * \param index The cell index inside the VectorSRMState.
		 * \param State Cell current state.
		 * \param CurrentTime Current simulation time.
		 *
		 * \return True if an spike is fired in the defined time. False in otherwise.
		 */
		bool CheckSpikeAt(int index, VectorSRMState * State, double CurrentTime);

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
		 * \brief It abstracts the effect of an input spike in the cell.
		 *
		 * It abstracts the effect of an input spike in the cell.
		 *
		 * \param index The cell index inside the VectorSRMState.
		 * \param State Cell current state.
		 * \param InputConnection Input connection from which the input spike has got the cell.
		 */
		virtual void SynapsisEffect(int index, VectorSRMState * State, Interconnection * InputConnection);

	public:
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object.
		 *
		 * \param NeuronTypeID Neuron type identificator
		 * \param NeuronModelID Neuron model identificator.
		 */
		SRMDetTimeDrivenModel(string NeuronTypeID, string NeuronModelID);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~SRMDetTimeDrivenModel();

		/*!
		 * \brief It loads the neuron model description and tables (if necessary).
		 *
		 * It loads the neuron model description and tables (if necessary).
		 */
		virtual void LoadNeuronModel() throw (EDLUTFileException);

		/*!
		 * \brief It return the Neuron Model VectorNeuronState 
		 *
		 * It return the Neuron Model VectorNeuronState 
		 *
		 */
		virtual VectorNeuronState * InitializeState();

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
		virtual InternalSpike * ProcessInputSpike(PropagatedSpike *  InputSpike);

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

		/*!
		 * \brief It prints the time-driven model info.
		 *
		 * It prints the current time-driven model characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out);

		/*!
		 * \brief It gets the neuron model type (event-driven or time-driven).
		 *
		 * It gets the neuron model type (event-driven or time-driven).
		 *
		 * \return The type of the neuron model.
		 */
		enum NeuronModelType GetModelType();

		/*!
		 * \brief It initialice VectorNeuronState.
		 *
		 * It initialice VectorSRMState.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 */
		void InitializeStates(int N_neurons);

};

#endif /* SRMDETTIMEDRIVENMODEL_H_ */
