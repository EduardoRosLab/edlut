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
	private:
	
		/*!
   		 * Activity register.
   		 */
   		float lpar[3];
   		
   		/*!
   		 * Activity register.
   		 */
   		float cpar[3];
	
	public:
	
		/*!
		 * \brief It gets the Lpar parameter of the learning rule.
		 * 
		 * It gets the Lpar parameter of the learning rule.
		 * 
		 * \param index Parameter index.
		 * 
		 * \return Lpar parameter at indexth position.
		 */
   		float GetLparAt(int index) const;
   		
   		/*!
		 * \brief It sets the Lpar parameter of the learning rule.
		 * 
		 * It sets the Lpar parameter of the learning rule.
		 * 
		 * \param index Parameter index.
		 * \param NewLpar parameter at indexth position.
		 */
   		void SetLparAt(int index, float NewLpar);
   		
   		/*!
		 * \brief It gets the Cpar parameter of the learning rule.
		 * 
		 * It gets the Cpar parameter of the learning rule.
		 * 
		 * \param index Parameter index.
		 * 
		 * \return Cpar parameter at indexth position.
		 */
   		float GetCparAt(int index) const;
   		
   		/*!
		 * \brief It sets the Cpar parameter of the learning rule.
		 * 
		 * It sets the Cpar parameter of the learning rule.
		 * 
		 * \param index Parameter index.
		 * \param NewCpar parameter at indexth position.
		 */
   		void SetCparAt(int index, float NewCpar);
	
		/*!
		 * \brief It gets the number of state variables that this learning rule needs.
		 * 
		 * It gets the number of state variables that this learning rule needs.
		 * 
		 * \return The number of state variables that this learning rule needs.
		 */
   		virtual int GetNumberOfVar() const;
   	
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
