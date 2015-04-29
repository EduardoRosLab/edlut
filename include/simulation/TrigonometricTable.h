/***************************************************************************
 *                           TrigonometricTable.h                          *
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

#ifndef TRIGONOMETRICTABLE_H_
#define TRIGONOMETRICTABLE_H_

#include <cmath>

/*!
 * \file TrigonometricTable.h
 *
 * \author Francisco Naveros
 * \date February 2015
 *
 * This file declares a look-up table for a sinusoidal and cosenoidal function.
 */
 


class TrigonometricTable{
	
   	public:
   		
		/*!
		 * Precalculated sin and cos terms.
		 */
		static float * TrigonometricLUT;

		/*!
		 * Precalculated LUT Step.
		 */
		static const float LUTStep;
		static const float inv_LUTStep;

		static const int N_ELEMENTS=1024*1024;



		/*!
		 * \brief It precompute a look-up table of sin and cos.
		 */
		static float * GenerateTrigonometricLUT(){
			float * NewSinLUT=new float[N_ELEMENTS*2];

			for (unsigned int i=0; i<N_ELEMENTS; ++i){
				NewSinLUT[2*i] = sinf(LUTStep*i);
				NewSinLUT[2*i+1] = cosf(LUTStep*i);
			}

			return NewSinLUT;
		}


		static int CalculateOffsetPosition(float ElapsedRelative){
			return ((int)(ElapsedRelative*inv_LUTStep + 0.5f))*2;
		}

		static float CalculateValidPosition(int initPosition, int offset){
			return((initPosition+offset)%(N_ELEMENTS*2));
		}

		static float GetElement(int index){
			return TrigonometricLUT[index];
		}
  	
};


#endif /*EXPONENTIALTABLE_H_*/



