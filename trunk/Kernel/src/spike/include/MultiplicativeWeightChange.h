/***************************************************************************
 *                           MultiplicativeWeightChange.h                  *
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

#ifndef MULTIPLICATIVEWEIGHTCHANGE_H_
#define MULTIPLICATIVEWEIGHTCHANGE_H_

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
 * \class MultiplicativeWeightChange
 *
 * \brief Multiplicative learning rule.
 *
 * This class abstract the behaviour of a multiplicative learning rule.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */ 
class MultiplicativeWeightChange: public WeightChange{
	
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


#endif /*MULTIPLICATIVEWEIGHTCHANGE_H_*/
