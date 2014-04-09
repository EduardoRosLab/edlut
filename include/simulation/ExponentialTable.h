/***************************************************************************
 *                           ExponentialTable.h                            *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
 * email                : fnaveros@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef EXPONENTIALTABLE_H_
#define EXPONENTIALTABLE_H_

#include <cmath>

/*!
 * \file ExponentialTable.h
 *
 * \author Francisco Naveros
 * \date November 2013
 *
 * This file declares a look-up table for an exponential function.
 */
 


class ExponentialTable{
	
   	public:
   		
   		/*!
   		 * Minimun value of the exponent.
   		 */
		float Min;

		/*!
   		 * Maximun value of the exponent.
   		 */
		float Max;

		/*!
   		 * Number of look-up table elements.
   		 */
		int TableSize;

		/*!
   		 * Look-up table.
   		 */
		float * LookUpTable;

		float aux;

   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes the look-up table.
   		 */
   		ExponentialTable(float minimun, float maximun, int size);
   	
		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~ExponentialTable();


   	
   		/*!
   		 * \brief It gets the result for the value.
   		 * 
   		 * It gets the result for the value.
   		 * 
   		 * \return the result for the value.
   		 */
   		float GetResult(float value);


   		

   	
};

#endif /*EXPONENTIALTABLE_H_*/
