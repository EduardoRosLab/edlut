/***************************************************************************
 *                           AdditiveWeightChange.h                        *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
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

#ifndef ADDITIVEWEIGHTCHANGE_H_
#define ADDITIVEWEIGHTCHANGE_H_

/*!
 * \file MultiplicativeWeightChange.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a multiplicative learning rule.
 */
 
#include "./WeightChange.h"
 
/*!
 * \class AdditiveWeightChange
 *
 * \brief Additive learning rule.
 *
 * This class abstract the behaviour of a additive learning rule.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */ 
class AdditiveWeightChange: public WeightChange{
	private:
		/*!
   		 * \brief It updates the previous activity in the connection.
   		 * 
   		 * It updates the previous activity in the connection.
   		 * 
   		 * \param time The spike time.
   		 * \param Connection The connection to be modified.
   		 * \param spike True if an spike is produced.
   		 */
		virtual void update_activity(double time,Interconnection * Connection,bool spike);
	
	public:
   	
   		/*!
   		 * \brief It applys the weight change function.
   		 * 
   		 * It applys the weight change function.
   		 * 
   		 * \param Connection The connection where the spike happened.
   		 * \param SpikeTime The spike time.
   		 */
   		virtual void ApplyWeightChange(Interconnection * Connection,double SpikeTime);
   		
   		/*!
		 * \brief It gets the number of state variables that this learning rule needs.
		 * 
		 * It gets the number of state variables that this learning rule needs.
		 * 
		 * \return The number of state variables that this learning rule needs.
		 */
   		virtual int GetNumberOfVar() const;
   		
};

#endif /*ADDITIVEWEIGHTCHANGE_H_*/
