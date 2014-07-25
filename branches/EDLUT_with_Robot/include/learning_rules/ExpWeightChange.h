/***************************************************************************
 *                           ExpWeightChange.h                             *
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

#ifndef EXPWEIGHTCHANGE_H_
#define EXPWEIGHTCHANGE_H_

/*!
 * \file ExpWeightChange.h
 *
 * \author Jesus Garrido
 * \date October 2011
 *
 * This file declares a class which abstracts a exponential additive learning rule.
 */
 
#include "./AdditiveKernelChange.h"
 
/*!
 * \class ExpWeightChange
 *
 * \brief Exponential learning rule.
 *
 * This class abstract the behaviour of a exponential additive learning rule.
 *
 * \author Jesus Garrido
 * \date October 2011
 */ 
class ExpWeightChange: public AdditiveKernelChange{
	private:
		

	public:
		/*!
		 * \brief Default constructor. It creates a new exponential additive learning-rule.
		 * 
		 * Default constructor. It creates a new exponential additive learning-rule
		 */
		ExpWeightChange();

		/*!
		 * \brief Object destructor.
		 *
		 * It remove the object.
		 */
		virtual ~ExpWeightChange();

		/*!
		 * \brief It initialize the state associated to the learning rule for all the synapses.
		 *
		 * It initialize the state associated to the learning rule for all the synapses.
		 *
		 * \param NumberOfSynapses the number of synapses that implement this learning rule.
		 */
		void InitializeConnectionState(unsigned int NumberOfSynapses);
};

#endif /*EXPWEIGHTCHANGE_H_*/
