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
 
#include "./AdditiveKernelChange.h"
 
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
class SinWeightChange: public AdditiveKernelChange{
	private:
	
		/*!
		 * The exponent of the sinuidal function.
		 */
		int exponent;
		
	public:
		/*!
		 * \brief Default constructor. It creates a new sinusoidal additive learning-rule
		 * with the parameter exponent.
		 * 
		 * Default constructor. It creates a new sinusoidal additive learning-rule
		 * with the parameter exponent.
		 */
		SinWeightChange();

		/*!
		 * \brief It gets the initial state associated to the learning rule.
		 *
		 * It gets the initial state associated to the learning rule.
		 *
		 * \return The initial state that the learning rule needs.
		 */
		virtual ConnectionState * GetInitialState();

		/*!
		 * \brief It loads the learning rule properties.
		 *
		 * It loads the learning rule properties.
		 *
		 * \param fh A file handler placed where the Learning rule properties are defined.
		 * \param Currentline The file line where the handler is placed.
		 *
		 * \throw EDLUTFileException If something wrong happens in reading the learning rule properties.
		 */
		virtual void LoadLearningRule(FILE * fh, long & Currentline) throw (EDLUTFileException);

	
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
