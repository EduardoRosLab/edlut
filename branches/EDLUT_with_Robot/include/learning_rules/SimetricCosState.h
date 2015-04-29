/***************************************************************************
 *                           CosState.h                                    *
 *                           -------------------                           *
 * copyright            : (C) 2014 by Niceto R. Luque& Francisco Naveros   *
 * email                : nlque@ugr.es, fnaveros@ugr.es                    *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef SIMETRICCOSSTATE_H_
#define SIMETRICCOSSTATE_H_

#include "ConnectionState.h"

/*!
 * \file CosState.h
 *
 * \author Niceto R. Luque Francisco Naveros
 * \date May 2014
 *
 * This file declares a class which abstracts the current state of a synaptic connection
 * with Cosinusoidal learning rule WITHOUT inner delay to be compensated.
 */

/*!
 * \class CosState
 *
 * \brief Synaptic connection current state.
 *
 * This class abstracts the state of a synaptic connection including Sinusoidal learning rule and defines the state variables of
 * that connection. The kernel function is f(t) = exp(-t/tau)*Cos(-t/tau)^2, where t represents the time since
 * the last presynaptic spike reached the cell.
 *
 * \author Niceto R. Luque Francisco Naveros
 * \date May 2014
 */

class SimetricCosState : public ConnectionState{

	private:
		

		/*!
		 * \brief Kernel amplitude in second.
		 */
		float tau;
		float inv_tau;

		/*!
		 * \brief Exponent
		 */
		float exponent;


	public:

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new state of a connection.
		 *
		 * \param NumSynapses Number of synapses that implement this learning rule.
		 * \param NewExponent The exponent of the sinusoidal function.
		 * \param MaxPosition Temporal position of the peak.
		 */
		SimetricCosState(unsigned int NumSynapses, float NewTau, float NewExponent);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~SimetricCosState();

		/*!
		 * \brief It gets the value of the accumulated presynaptic activity.
		 *
		 * It gets the value of the accumulated presynaptic activity.
		 *
		 * \return The accumulated presynaptic activity.
		 */
		virtual float GetPresynapticActivity(unsigned int index);

		/*!
		 * \brief It gets the value of the accumulated postsynaptic activity.
		 *
		 * It gets the value of the accumulated postsynaptic activity.
		 *
		 * \return The accumulated postsynaptic activity.
		 */
		virtual float GetPostsynapticActivity(unsigned int index);


		/*!
		 * \brief It gets the number of variables that you can print in this state.
		 *
		 * It gets the number of variables that you can print in this state.
		 *
		 * \return The number of variables that you can print in this state.
		 */
		virtual unsigned int GetNumberOfPrintableValues();

		/*!
		 * \brief It gets a value to be printed from this state.
		 *
		 * It gets a value to be printed from this state.
		 *
		 * \return The value at position-th position in this state.
		 */
		virtual double GetPrintableValuesAt(unsigned int position);


		/*!
		 * \brief set new time to spikes.
		 *
		 * It set new time to spikes.
		 *
		 * \param index The synapse's index inside the learning rule.
		 * \param NewTime new time.
		 * \param pre_post In some learning rules (i.e. STDPLS) this variable indicate wether the update affects the pre- or post- variables.
		 */
		virtual void SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post);


		/*!
		 * \brief It implements the behaviour when it transmits a spike.
		 *
		 * It implements the behaviour when it transmits a spike. It must be implemented
		 * by any inherited class.
		 *
		 * \param index The synapse's index inside the learning rule.
		 */
		virtual void ApplyPresynapticSpike(unsigned int index);

		/*!
		 * \brief It implements the behaviour when the target cell fires a spike.
		 *
		 * It implements the behaviour when it the target cell fires a spike. It must be implemented
		 * by any inherited class.
		 *
		 * \param index The synapse's index inside the learning rule.
		 */
		virtual void ApplyPostsynapticSpike(unsigned int index);

};

#endif /* NEURONSTATE_H_ */

