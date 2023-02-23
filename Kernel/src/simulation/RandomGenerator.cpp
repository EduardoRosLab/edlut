/***************************************************************************
 *                           RandomGenerator.cpp	                       *
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


#include "../../include/simulation/RandomGenerator.h"


unsigned long int RandomGenerator::global_seed = 1;


RandomGenerator::RandomGenerator(){
	this->next_element = this->global_seed;
	generate_next_seed();
}

RandomGenerator::~RandomGenerator(){

}