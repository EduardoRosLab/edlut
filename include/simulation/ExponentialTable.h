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
		static const float Min;

		/*!
   		 * Maximun value of the exponent.
   		 */
		static const float Max;

		/*!
   		 * Number of look-up table elements.
   		 */
		static const int TableSize=1024*1024*4;

		/*!
   		 * Look-up table.
   		 */
		static float * LookUpTable;

		static const float aux;

		static float * generate_data(){
			float * NewLookUpTable=new float[TableSize];
			for(int i=0; i<TableSize; i++){
				float exponent = Min + ((Max-Min)*i)/(TableSize-1);
				NewLookUpTable[i]=exp(exponent);		
			}
			return NewLookUpTable;
		}
   	
   		/*!
   		 * \brief It gets the result for the value.
   		 * 
   		 * It gets the result for the value.
   		 * 
   		 * \return the result for the value.
   		 */
		static float GetResult(float value){
			if(value>=Min && value<=Max){
				int position=(value-Min)*aux;
				return LookUpTable[position];
			}else{
				if(value<(Min)){
					return 0.0f;
				}else{
					return exp(value);
				}
			}
		} 		

   	
};





#endif /*EXPONENTIALTABLE_H_*/
