/***************************************************************************
 *                           SimetricCosSTDPState.h                        *
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

#ifndef SIMETRICCOSSTDPSTATE_H_
#define SIMETRICCOSSTDPSTATE_H_

#include "ConnectionState.h"

/*!
 * \file SimetricCosSTDPState.h
 *
 * \author Jesus Garrido
 * \date October 2011
 *
 * This file declares a class which abstracts the current state of a synaptic connection
 * with STDP capabilities.
 */

/*!
 * \class STDPState
 *
 * \brief Synaptic connection current state.
 *
 * This class abstracts the state of a synaptic connection including STDP and defines the state variables of
 * that connection.
 *
 * \author Jesus Garrido
 * \date October 2011
 */

class SimetricCosSTDPState : public ConnectionState{

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
		 * \param LTPtau Time constant of the LTP component.
		 * \param LTDtau Time constant of the LTD component.
		 */
		SimetricCosSTDPState(int NumSynapsesAndNeurons, float NewTau, float NewExponent);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~SimetricCosSTDPState();

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
		 * \param index The synapse's index inside the learning rule.
		 * \param position Position inside each connection.
		 *
		 * \return The value at position-th position in this state.
		 */
		virtual double GetPrintableValuesAt(unsigned int index, unsigned int position);


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

		/*!
		 * \brief It initialize the synaptic weight and max synaptic weight in the learning rule (required for some specifics learning rules).
		 *
		 * It initialize the synaptic weight and max synaptic weight in the learning rule (required for some specifics learning rules).
		 *
		 * \param index The synapse's index inside the learning rule.
		 * \param weight synaptic weight
		 * \param max_weight max synaptic weight
		 */
		virtual void SetWeight(unsigned int index, float weight, float max_weight){}

};

#endif /* SIMETRICCOSSTDPSTATE_H_ */
