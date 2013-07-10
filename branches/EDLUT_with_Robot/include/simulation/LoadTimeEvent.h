/***************************************************************************
 *                           LoadTimeEvent.h                               *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Francisco Naveros                    *
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

#ifndef LOADTIMEEVENT_H_
#define LOADTIMEEVENT_H_

/*!
 * \file LoadTimeEvent.h
 *
 * \author Francisco Naveros
 * \date October 2012
 *
 * This file declares a class which load time driven nueron step for a CPU.
 */

#include <string>
using namespace std;

#include "../../include/simulation/Utils.h"
#include "../../include/simulation/Configuration.h"




/*!
 * \class LoadTimeEvent
 *
 * \brief Load TimeEvent
 *
 * \author Francisco Naveros
 * \date October 2012
 */
class LoadTimeEvent {
	protected:

	public:

		static double loadTimeEvent(FILE *fh, long * Currentline)throw (EDLUTFileException){
			double time_step=0.0;

			skip_comments(fh,*Currentline);
			if(fscanf(fh,"%lf",&time_step)==1){
				if(time_step<=0.0){
////NEW CODE------------------------------------------------------------------------------
					throw EDLUTFileException(4,7,6,1,*Currentline);
////--------------------------------------------------------------------------------------
				}
			}else{
//NEW CODE------------------------------------------------------------------------------
				throw EDLUTFileException(4,7,6,1,*Currentline);
//--------------------------------------------------------------------------------------
			}

			return time_step;
		}
};




#endif /* LOADTIMEEVENT_H_ */
