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


RealTimeRestriction::RealTimeRestriction(): slot_time(0.0f), max_delay(0.0f), first_section(1.0f), second_section(1.0f), third_section(1.0f),
		Stop(false), RestrictionLevel(0){
}

RealTimeRestriction::RealTimeRestriction(float new_slot_time, float new_max_delay, float new_first_section, float new_second_section, float new_third_section): slot_time(new_slot_time), 
		Stop(false), RestrictionLevel(0), init_compensation_factor(1.0f){

	if(new_max_delay<new_slot_time){
		max_delay=new_slot_time;
	}else{
		max_delay=new_max_delay;
	}
	N_elements=max_delay/new_slot_time;
	max_delay=N_elements*new_slot_time;

	slot_elapsed_times=new float[N_elements]();


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

	position=N_elements-1;

	first_gap_time=max_delay*(1.0f-first_section);
	second_gap_time=max_delay*(1.0f-second_section);
	third_gap_time=max_delay*(1.0f-third_section);


	#if defined(REAL_TIME_WINNT)
		if(!QueryPerformanceFrequency(&this->freq))
			puts("QueryPerformanceFrequency failed");
	#elif defined (REAL_TIME_LINUX)
		if(clock_getres(CLOCK_REALTIME, &this->freq))
			puts("clock_getres failed");
	#elif defined (REAL_TIME_OSX)
		// If this is the first time we've run, get the timebase.
		// We can use denom == 0 to indicate that sTimebaseInfo is
		// uninitialised because it makes no sense to have a zero
		// denominator is a fraction.
		if (this->freq.denom == 0 ) {
			(void) mach_timebase_info(&this->freq);
		}
	#endif

		#if defined(REAL_TIME_WINNT)
   			QueryPerformanceCounter(&startt);
		#elif defined(REAL_TIME_LINUX)
    		clock_gettime(CLOCK_REALTIME, &startt);
		#elif defined(REAL_TIME_OSX)
			startt = mach_absolute_time();
		#endif
}
   	
  		
RealTimeRestriction::~RealTimeRestriction(){
}

void RealTimeRestriction::SetParameterWatchDog(float new_slot_time, float new_max_delay, float new_first_section, float new_second_section, float new_third_section){
	init_compensation_factor=1.0f;
	position=0;
	slot_time=new_slot_time;

	if(new_max_delay<new_slot_time){
		max_delay=new_slot_time;
	}else{
		max_delay=new_max_delay;
	}
	N_elements=max_delay/new_slot_time;
	max_delay=N_elements*new_slot_time;

	slot_elapsed_times=new float[N_elements]();


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


	position=N_elements-1;

	first_gap_time=max_delay*(1.0f-first_section);
	second_gap_time=max_delay*(1.0f-second_section);
	third_gap_time=max_delay*(1.0f-third_section);


	#if defined(REAL_TIME_WINNT)
		if(!QueryPerformanceFrequency(&this->freq))
			puts("QueryPerformanceFrequency failed");
	#elif defined (REAL_TIME_LINUX)
		if(clock_getres(CLOCK_REALTIME, &this->freq))
			puts("clock_getres failed");
	#elif defined (REAL_TIME_OSX)
		// If this is the first time we've run, get the timebase.
		// We can use denom == 0 to indicate that sTimebaseInfo is
		// uninitialised because it makes no sense to have a zero
		// denominator is a fraction.
		if (this->freq.denom == 0 ) {
			(void) mach_timebase_info(&this->freq);
		}
	#endif

		#if defined(REAL_TIME_WINNT)
   			QueryPerformanceCounter(&startt);
		#elif defined(REAL_TIME_LINUX)
    		clock_gettime(CLOCK_REALTIME, &startt);
		#elif defined(REAL_TIME_OSX)
			startt = mach_absolute_time();
		#endif
}
 


void RealTimeRestriction::ResetWatchDog(){
float current_time_supervision;
	#if defined(REAL_TIME_WINNT)
		QueryPerformanceCounter(&endt_supervision); // measures time
		current_time_supervision=(endt_supervision.QuadPart-startt.QuadPart)/(float)this->freq.QuadPart; // to be logged
	#elif defined(REAL_TIME_LINUX)
		clock_gettime(CLOCK_REALTIME, &endt_supervision);
		// Calculate time it took
		 current_time_supervision = (endt_supervision.tv_sec-startt.tv_sec ) + (endt_supervision.tv_nsec-startt.tv_nsec )/float(1e9);
	#elif defined(REAL_TIME_OSX)
		// Stop the clock.
		endt_supervision = mach_absolute_time();
		// Calculate the duration.
		elapsed_supervision = endt_supervision - startt;
		current_time_supervision = 1e-9 * elapsed_supervision * this->freq.numer / this->freq.denom;
	#endif


	float portion=max_delay/N_elements;

	for(int j=0; j<N_elements; j++){
		slot_elapsed_times[j]=current_time_supervision+((-N_elements+1+j)*portion);
	}
	position=N_elements-1;
	init_compensation_factor=1;
}

void RealTimeRestriction::NextStepWatchDog(){
			float current_time_supervision;
			position=(position+1)%N_elements;

			#if defined(REAL_TIME_WINNT)
				QueryPerformanceCounter(&endt_supervision); // measures time
				current_time_supervision=(endt_supervision.QuadPart-startt.QuadPart)/(float)this->freq.QuadPart; // to be logged
			#elif defined(REAL_TIME_LINUX)
				clock_gettime(CLOCK_REALTIME, &endt_supervision);
				// Calculate time it took
				 current_time_supervision = (endt_supervision.tv_sec-startt.tv_sec ) + (endt_supervision.tv_nsec-startt.tv_nsec )/float(1e9);
			#elif defined(REAL_TIME_OSX)
				// Stop the clock.
				endt_supervision = mach_absolute_time();
				// Calculate the duration.
				elapsed_supervision = endt_supervision - startt;
				current_time_supervision = 1e-9 * elapsed_supervision * this->freq.numer / this->freq.denom;
			#endif
			float limit1=current_time_supervision+max_delay;

			float limit2=slot_elapsed_times[position]+max_delay;

			if(limit1<=limit2){
				slot_elapsed_times[position]=limit1;
			}else{
				slot_elapsed_times[position]=limit2;
			}

			if(init_compensation_factor<N_elements){
				init_compensation_factor+=1;
			}

			time=slot_elapsed_times[position];
		float gap=((time-current_time_supervision)/init_compensation_factor)*N_elements;
		if(gap>first_gap_time){
			RestrictionLevel=0;
		}else if(gap>second_gap_time){
			RestrictionLevel=1;
		}else if(gap>third_gap_time){
			RestrictionLevel=2;
		}else{
			RestrictionLevel=3;
		}
}


void RealTimeRestriction::StopWatchDog(){
	Stop=true;
}

void RealTimeRestriction::Watchdog(){
	if(this->slot_time=0.0f){
		puts("Real time restriction object parameter not initialized");
	}

	float current_time;

	while(true){
		if(Stop==true){
			break;
		}

		#if defined(REAL_TIME_WINNT)
			QueryPerformanceCounter(&endt); // measures time
			current_time=(endt.QuadPart-startt.QuadPart)/(float)this->freq.QuadPart; // to be logged
		#elif defined(REAL_TIME_LINUX)
			clock_gettime(CLOCK_REALTIME, &endt);
			// Calculate time it took
			 current_time = (endt.tv_sec-startt.tv_sec ) + (endt.tv_nsec-startt.tv_nsec )/float(1e9);
		#elif defined(REAL_TIME_OSX)
			// Stop the clock.
			endt = mach_absolute_time();
			// Calculate the duration.
			elapsed = endt - startt;
			current_time = 1e-9 * elapsed * this->freq.numer / this->freq.denom;
		#endif


		float gap=((time-current_time)/init_compensation_factor)*N_elements;
		if(gap>first_gap_time){
			RestrictionLevel=0;
		}else if(gap>second_gap_time){
			RestrictionLevel=1;
		}else if(gap>third_gap_time){
			RestrictionLevel=2;
		}else{
			RestrictionLevel=3;
		}
	}
}














//void RealTimeRestriction::ResetWatchDog(){
//	this->Reset=true;
//	this->RestrictionLevel=0;
//}
//
//void RealTimeRestriction::NextStepWatchDog(){
//	this->NextStep=true;
//	//this->RestrictionLevel=0;
//}
//
//
//void RealTimeRestriction::StopWatchDog(){
//	Stop=true;
//}
//
//void RealTimeRestriction::Watchdog(){
//	if(this->slot_time=0.0f){
//		puts("Real time restriction object parameter not initialized");
//	}
//
//	float current_time;
//
//	while(true){
//		if(Stop==true){
//			break;
//		}
//
//		if(Reset==true){
//			Reset=false;
//			#if defined(REAL_TIME_WINNT)
//   				QueryPerformanceCounter(&startt);
//			#elif defined(REAL_TIME_LINUX)
//    			clock_gettime(CLOCK_REALTIME, &startt);
//			#elif defined(REAL_TIME_OSX)
//				startt = mach_absolute_time();
//			#endif
//
//
//float portion=max_delay/N_elements;
//for(int j=0; j<N_elements; j++){
////	slot_elapsed_times[j]=(j+1)*portion;
//	slot_elapsed_times[j]=0.0f;
//}
//			position=N_elements-1;
//		}
//
//		if(NextStep==true){
//			NextStep=false;
//			position=(position+1)%N_elements;
//
//			float limit1=slot_elapsed_times[position]+max_delay;
//
//			#if defined(REAL_TIME_WINNT)
//				QueryPerformanceCounter(&endt); // measures time
//				current_time=(endt.QuadPart-startt.QuadPart)/(float)this->freq.QuadPart; // to be logged
//			#elif defined(REAL_TIME_LINUX)
//				clock_gettime(CLOCK_REALTIME, &endt);
//				// Calculate time it took
//				 current_time = (endt.tv_sec-startt.tv_sec ) + (endt.tv_nsec-startt.tv_nsec )/float(1e9);
//			#elif defined(REAL_TIME_OSX)
//				// Stop the clock.
//				endt = mach_absolute_time();
//				// Calculate the duration.
//				elapsed = endt - startt;
//				current_time = 1e-9 * elapsed * this->freq.numer / this->freq.denom;
//			#endif
//			float limit2=current_time+max_delay;
//
//			if(limit1<=limit2){
//				slot_elapsed_times[position]=limit1;
//			}else{
//				slot_elapsed_times[position]=limit2;
//			}
//		}
//
//		if(Reset==false && NextStep==false){
//			#if defined(REAL_TIME_WINNT)
//				QueryPerformanceCounter(&endt); // measures time
//				current_time=(endt.QuadPart-startt.QuadPart)/(float)this->freq.QuadPart; // to be logged
//			#elif defined(REAL_TIME_LINUX)
//				clock_gettime(CLOCK_REALTIME, &endt);
//				// Calculate time it took
//				 current_time = (endt.tv_sec-startt.tv_sec ) + (endt.tv_nsec-startt.tv_nsec )/float(1e9);
//			#elif defined(REAL_TIME_OSX)
//				// Stop the clock.
//				endt = mach_absolute_time();
//				// Calculate the duration.
//				elapsed = endt - startt;
//				current_time = 1e-9 * elapsed * this->freq.numer / this->freq.denom;
//			#endif
//
//
//			if((slot_elapsed_times[position]-current_time)>first_gap_time){
//				RestrictionLevel=0;
//			}else if((slot_elapsed_times[position]-current_time)<second_gap_time){
//				RestrictionLevel=1;
//			}else if((slot_elapsed_times[position]-current_time)<third_gap_time){
//				RestrictionLevel=2;
//			}else{
//				RestrictionLevel=3;
//			}
//		}
//	}
//}
