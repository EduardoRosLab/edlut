/***************************************************************************
 *                           SimetricCosSinSTDPState.h                     *
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

#ifndef SIMETRICCOSSINSTDPSTATE_H_
#define SIMETRICCOSSINSTDPSTATE_H_

#include "ConnectionState.h"

/*!
 * \file SimetricCosSinSTDPState.h
 *
 * \author Francisco Naveros
 * \date November 2014
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

class SimetricCosSinSTDPState : public ConnectionState{

	private:
		

		/*!
		 * Distance un second between the maximun and minimun value
		 */
		float MaxMinDistance;
		float inv_MaxMinDistance;

		
		/*!
		 * Auxiliar value used to store the sin function period.
		 */
		float inv_AuxMaxMinDistanceForSin;

		/*!
		 * Max increment in nanosiemen for the central lobule
		 */
		float CentralAmplitudeFactor;

		/*!
		 * Max increment in nanosiemen for the lateral lobules
		 */
		float LateralAmplitudeFactor;



		/*!
		 * Exponential factor for the central lobule
		 */
		const float CentralExpFactor;

		/*!
		 * Exponential factor for the lateral lobules
		 */
		const float LateralExpFactor;


public:

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new state of a connection.
		 *
		 * \param NumSynapses Number of synapses that implement this learning rule.
		 * \param NewMaxMinDistancen distance un second between the maximun and minimun value.
		 * \param NewCentralAmplitudeFactor Max increment in nanosiemen for the central lobule.
		 * \param NewLateralAmplitudeFactor Max increment in nanosiemen for the lateral lobules.
		 */
		SimetricCosSinSTDPState(unsigned int NumSynapses, float NewMaxMinDistance, float NewCentralAmplitudeFactor, float NewLateralAmplitudeFactor);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~SimetricCosSinSTDPState();

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

