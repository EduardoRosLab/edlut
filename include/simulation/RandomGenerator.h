/***************************************************************************
 *                           RandomGenerator.h	                           *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef RANDOMGENERATOR_H_
#define RANDOMGENERATOR_H_

#include <cmath>

/*!
 * \file RandomGenerator.h
 *
 * \author Francisco Naveros
 * \date April 2015
 *
 * This file declares a random generator.
 */
 


class RandomGenerator{
	
   	public:
   		
	static const int MAX_RAND=2147483647;//(2^31)-1 
	static unsigned long int next_element;  	

	static unsigned int rand(void){
		next_element = ((next_element * 1103515245) + 12345) & 0x7fffffff;
		return (unsigned int)next_element;
	}

	static float frand(void){
		next_element = ((next_element * 1103515245) + 12345) & 0x7fffffff;
		return ((float)next_element)/MAX_RAND;
	}
	
	static double drand(void){
		next_element = ((next_element * 1103515245) + 12345) & 0x7fffffff;
		return ((double)next_element)/MAX_RAND;
	}


	static void srand(unsigned int seed){
		next_element = seed;
	}
};



#endif /*RANDOMGENERATOR_H_*/



