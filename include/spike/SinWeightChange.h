/***************************************************************************
 *                           SinWeightChange.h                        *
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

#ifndef SINWEIGHTCHANGE_H_
#define SINWEIGHTCHANGE_H_

/*!
 * \file SinWeightChange.h
 *
 * \author Jesus Garrido
 * \author Niceto Luque
 * \author Richard Carrillo
 * \date July 2009
 *
 * This file declares a class which abstracts a exponential-sinuidal additive learning rule.
 */
 
#include "./AdditiveWeightChange.h"
 
/*!
 * \class SinWeightChange
 *
 * \brief Sinuidal learning rule.
 *
 * This class abstract the behaviour of a exponential-sinuidal additive learning rule.
 *
 * \author Jesus Garrido
 * \author Niceto Luque
 * \author Richard Carrillo
 * \date July 2009
 */ 
class SinWeightChange: public AdditiveWeightChange{
	private:
		/*!
		 * Precalculated terms.
		 */
	    const static float terms [11][11];
	
		/*!
		 * The exponent of the sinuidal function.
		 */
		int exponent;
		
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
		 * \brief Default constructor. It creates a new sinusoidal additive learning-rule
		 * with the parameter exponent.
		 * 
		 * Default constructor. It creates a new sinusoidal additive learning-rule
		 * with the parameter exponent.
		 * \param NewExponent Exponent for the learning rule. e^(-x)*Sin(x)^Exponent.
		 */
		SinWeightChange(int NewExponent);
	
		/*!
		 * \brief It gets the number of state variables that this learning rule needs.
		 * 
		 * It gets the number of state variables that this learning rule needs.
		 * 
		 * \return The number of state variables that this learning rule needs.
		 */
   		virtual int GetNumberOfVar() const;
   		
   		/*!
		 * \brief It gets the value of the exponent in the sin function.
		 * 
		 * It gets the value of the exponent in the sin function.
		 * 
		 * \return The value of the exponent in the sin function.
		 */
   		int GetExponent() const;
   		
};

#endif /*SINWEIGHTCHANGE_H_*/
