/***************************************************************************
 *                           RealTimeRestriction.h                         *
 *                           -------------------                           *
 * copyright            : (C) 2014 by Francisco Naveros                    *
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

#include "../../include/simulation/RealTimeRestriction.h"

#include <stdio.h>


RealTimeRestriction::RealTimeRestriction(float new_slot_time, float new_first_section, float new_second_section, float new_third_section): slot_time(new_slot_time), 
		Reset(false), Stop(false), RestrictionLevel(0){

	if(new_third_section<=1 && new_third_section>=0){
		third_section=new_third_section;
	}else{
		third_section=1;
	}

	if(new_second_section<=third_section && new_second_section>=0){
		second_section=new_second_section;
	}else{
		second_section=third_section;
	}

	if(new_first_section<=second_section && new_first_section>=0){
		first_section=new_first_section;
	}else{
		first_section=second_section;
	}


	#if defined(REAL_TIME_WINNT)
		if(!QueryPerformanceFrequency(&freq))
			puts("QueryPerformanceFrequency failed");
	#elif defined (REAL_TIME_LINUX)
		if(clock_getres(CLOCK_REALTIME, &freq))
			puts("clock_getres failed");
	#elif defined (REAL_TIME_OSX)
		// If this is the first time we've run, get the timebase.
		// We can use denom == 0 to indicate that sTimebaseInfo is
		// uninitialised because it makes no sense to have a zero
		// denominator is a fraction.
		if (freq.denom == 0 ) {
			(void) mach_timebase_info(&freq);
		}
	#endif
}
   	
  		
RealTimeRestriction::~RealTimeRestriction(){
}
 

void RealTimeRestriction::ResetWatchDog(){
	this->Reset=true;
	this->RestrictionLevel=0;
}

void RealTimeRestriction::StopWatchDog(){
	Stop=true;
}

void RealTimeRestriction::Watchdog(){
	while(true){
		if(Stop==true){
			break;
		}

		if(Reset==true){
			Reset=false;
			#if defined(REAL_TIME_WINNT)
   				QueryPerformanceCounter(&startt);
			#elif defined(REAL_TIME_LINUX)
    			clock_gettime(CLOCK_REALTIME, &startt);
			#elif defined(REAL_TIME_OSX)
				startt = mach_absolute_time();
			#endif
		}

		#if defined(REAL_TIME_WINNT)
			QueryPerformanceCounter(&endt); // measures time
			slot_elapsed_time=(endt.QuadPart-startt.QuadPart)/(float)freq.QuadPart; // to be logged
		#elif defined(REAL_TIME_LINUX)
			clock_gettime(CLOCK_REALTIME, &endt);
			// Calculate time it took
			 slot_elapsed_time = (endt.tv_sec-startt.tv_sec ) + (endt.tv_nsec-endt.tv_nsec )/float(1e9);
		#elif defined(REAL_TIME_OSX)
			// Stop the clock.
			endt = mach_absolute_time();
			// Calculate the duration.
			elapsed = endt - startt;
			slot_elapsed_time = 1e-9 * elapsed * freq.numer / freq.denom;
		#endif

		if(slot_elapsed_time>(slot_time*third_section) && Reset==false){
			RestrictionLevel=3;
		}else if(slot_elapsed_time>(slot_time*second_section) && Reset==false){
			RestrictionLevel=2;
		}else if(slot_elapsed_time>(slot_time*first_section) && Reset==false){
			RestrictionLevel=1;
		}else{
			RestrictionLevel=0;
		}
	}
}
