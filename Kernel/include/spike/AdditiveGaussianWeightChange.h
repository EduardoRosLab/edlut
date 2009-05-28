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

#ifndef ADDITIVEGAUSSIANWEIGHTCHANGE_H_
#define ADDITIVEGAUSSIANWEIGHTCHANGE_H_

/*!
 * \file AdditiveGaussianWeightChange.h
 *
 * \author Jesus Garrido
 * \author Niceto Luque
 * \date May 2009
 *
 * This file declares a class which abstracts an additive gaussian learning rule.
 */
 
#include "./WeightChange.h"
 
/*!
 * \class AdditiveGaussianWeightChange
 *
 * \brief Additive gaussian learning rule.
 *
 * This class abstract the behaviour of a additive gaussian learning rule.
 *
 * \author Jesus Garrido
 * \author Niceto Luque
 * \date May 2009
 */ 
class AdditiveGaussianWeightChange: public WeightChange{
	private:
	
		/*!
   		 * \brief It gets the gaussian component in the activity.
   		 * 
   		 * It gets the gaussian component in the activity.
   		 * 
   		 * \param time The current time.
   		 * \param Connection The connection to be modified.
   		 */
		float GaussianValue(double time,Interconnection * Connection);
		
		/*!
   		 * \brief It updates the previous activity in the connection.
   		 * 
   		 * It updates the previous activity in the connection.
   		 * 
   		 * \param time The spike time.
   		 * \param Connection The connection to be modified.
   		 * \param spike True if an spike is produced.
   		 */
		void update_activity(double time,Interconnection * Connection,bool spike);
	
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
   		
};

#endif /*ADDITIVEWEIGHTCHANGE_H_*/
