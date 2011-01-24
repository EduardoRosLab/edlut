/***************************************************************************
 *                           SRMTimeDrivenModel.h                          *
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

#ifndef SRMTIMEDRIVENMODEL_H_
#define SRMTIMEDRIVENMODEL_H_

/*!
 * \file SRMTimeDrivenModel.h
 *
 * \author Jesus Garrido
 * \date January 2011
 *
 * This file declares a class which implements a SRM (Spike response model) time-driven neuron
 * model.
 */

#include "./TimeDrivenNeuronModel.h"

#include "../spike/EDLUTFileException.h"

class SRMState;
class BufferedState;
class Interconnection;

/*!
 * \class SRMTimeDrivenModel
 *
 * \brief Spike Response time-driven neuron model non-based in look-up tables.
 *
 * This class implements the behavior of a neuron in a spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect...).
 * This behavior is calculated based in a buffer of activity.
 *
 * \author Jesus Garrido
 * \date January 2010
 */
class SRMTimeDrivenModel: public TimeDrivenNeuronModel {

	private:
		/*!
		 * \brief Decay time constant of the EPSP
		 */
		float tau;

		/*!
		 * \brief EPSP Step time
		 */
		float EPSPStep;

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
		 * \brief Time-driven step time
		 */
		float steptime;

		/*!
		 * \brief EPSP pre-calculated vector
		 */
		double * EPSP;


	protected:
		/*!
		 * \brief It precalculates the EPSP values.
		 *
		 * It precalculates the EPSP values.
		 */
		void PrecalculateEPSP();

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
		double PotentialIncrement(SRMState * State, double CurrentTime);

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
		bool CheckSpikeAt(SRMState * State, double CurrentTime);

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
		 * \param State Cell current state.
		 * \param InputConnection Input connection from which the input spike has got the cell.
		 */
		virtual void SynapsisEffect(SRMState * State, Interconnection * InputConnection);

	public:
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object.
		 *
		 * \param NeuronModelID Neuron model identificator.
		 */
		SRMTimeDrivenModel(string NeuronModelID);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~SRMTimeDrivenModel();

		/*!
		 * \brief It loads the neuron model description and tables (if necessary).
		 *
		 * It loads the neuron model description and tables (if necessary).
		 */
		virtual void LoadNeuronModel() throw (EDLUTFileException);

		/*!
		 * \brief It initializes the neuron state to defined values.
		 *
		 * It initializes the neuron state to defined values.
		 *
		 * \param State Cell current state.
		 */
		virtual NeuronState * InitializeState();

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
		 * \param The current neuron state.
		 * \param CurrentTime Current time.
		 *
		 * \return True if an output spike have been fired. False in other case.
		 */
		virtual bool UpdateState(NeuronState * State, double CurrentTime);

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
};

#endif /* SRMMODEL_H_ */
