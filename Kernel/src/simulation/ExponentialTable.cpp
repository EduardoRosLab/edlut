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

#include <cmath>
#include <iostream>

#include "../../include/simulation/ExponentialTable.h"




ExponentialTable::ExponentialTable(float minimum, float maximum, int size): Min(minimum), Max(maximum), TableSize(size){
	LookUpTable=new float[TableSize];
	for(int i=0; i<TableSize; i++){
		float exponent = Min + ((Max-Min)*i)/(TableSize-1);
		LookUpTable[i]=exp(exponent);		
	}
aux=1.0f/(Max-Min);
}
   		
ExponentialTable::~ExponentialTable(){
	delete LookUpTable;
}
   	
float ExponentialTable::GetResult(float value){
	if(value>=Min && value<=Max){
		//int position=((value-Min)/(Max-Min))*(TableSize-1);
		int position=((value-Min)*aux)*(TableSize-1);
		return LookUpTable[position];
	}else{
		if(value<(-20)){
			return 0.0f;
		}else{
			return exp(value);
		}
	}
}


   		


