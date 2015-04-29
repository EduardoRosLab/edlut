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

#ifndef REALTIMERESTRICTION_H_
#define REALTIMERESTRICTION_H_

/*!
 * \file RealTimeRestriction.h
 *
 * \author Francisco Naveros
 * \date June 2014
 *
 * This file declares a class which control how much time is consumed in each slot time for the 
 * robot control. This class is used to implement the real time restriction.
 */
 
#if defined(__APPLE_CC__)
  // MAC OS X includes
#	define REAL_TIME_OSX
#elif defined (__linux__)
  // Linux includes
#	define REAL_TIME_LINUX
#elif (defined(_WIN32) || defined(_WIN64))
#	define REAL_TIME_WINNT
#else
#	error "Unsupported platform"
#endif


#if defined(REAL_TIME_OSX)
#	include <mach/mach.h>
#	include <mach/mach_time.h>
#	include <CoreServices/CoreServices.h>
#	include <unistd.h>
#elif defined(REAL_TIME_LINUX)
#	include <time.h>
#elif defined(REAL_TIME_WINNT)
#	include <windows.h>
#endif


#include <iostream>

using namespace std;





/*!
 * \class RealTimeRestriction
 *
 * \brief This class is used to implement the real time restriction.
 *
 * This class control how much time is consumed in each slot time for the robot control. This class implement
 * a watchdog which increment the value of the variable "RestrictionLevel" to control the priority of the
 * events that must be processes. It uses the variables "first_section", "second_section" and "third_section"
 * to divide the "slot_time" in four sections. When the "slot_elapsed_time" pass from one secction to another,
 * the RestrictionLevel increases. All the events in EDLUT use this resctriction level to know what they must
 * do. For instant, when the restriction level is higher than zero, the learning rules are ignores.
 *
 * \author Francisco Naveros
 * \date June 2014
 */
class RealTimeRestriction{
public:

	/*!
	 * Auxiliar variables to measure the time with a precission of nanosecon in differents Operative systemes. 
	 */
	#if defined(REAL_TIME_WINNT)
		// Variables for consumed-CPU-time measurement
		LARGE_INTEGER startt,endt,freq,endt_supervision;

	#elif defined(REAL_TIME_OSX)
		uint64_t startt, endt, elapsed,endt_supervision, elapsed_supervision;
		mach_timebase_info_data_t freq;
	
	#elif defined(REAL_TIME_LINUX)
		// Calculate time taken by a request - Link with real-time library -lrt
		struct timespec startt, endt, freq,endt_supervision;
	#endif

		/*!
		 * Restriction level fixed by the batchdog.
		 */
		 int RestrictionLevel;

		 float time;

		/*!
		 * Auxiliar value where the watch dog measures the consumed time after each reset.
		 */
		float * slot_elapsed_times;
		int position;

		 int init_compensation_factor;


		/*!
		 * It controls when the watchdogs must be stopped.
		 */
		 bool Stop;


		/*!
		 * This is the reference slot time. Each execution slot must consume less time that this slot time.
		 */
		float slot_time;

		float max_delay;

		int N_elements;

		/*!
		 * value between 0 and 1 that contain the fraction of slot_time that diffines the first section.
		 */
		float first_section;

		/*!
		 * value between "first_section" and 1 that contain the fraction of slot_time that diffines the second section.
		 */
		float second_section;

		/*!
		 * value between "second_section" and 1 that contain the fraction of slot_time that diffines the third section.
		 */
		float third_section;

		float first_gap_time;
		float second_gap_time;
		float third_gap_time;
	
   	public:
   		

		 /*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new event object.
		 *
   		 */
   		RealTimeRestriction();

   		/*!
   		 * It creates and initializes a new event object.
		 *
		 * \param new_slot_time
		 * \param new_max_delay
		 * \param new_first_section
		 * \param new_second_section
		 * \param new_third_section
   		 */
   		RealTimeRestriction(float new_slot_time, float new_max_delay, float new_first_section, float new_second_section, float new_third_section);
   	
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~RealTimeRestriction();


   		/*!
   		 * \brief It sets the watchdog parameter.
   		 * 
   		 * It sets the watchdog parameter.
   		 */
		void SetParameterWatchDog(float new_slot_time, float new_max_delay, float new_first_section, float new_second_section, float new_third_section);

   		/*!
   		 * \brief It resets the watchdog.
   		 * 
   		 * It resets the watchdog.
   		 */		
		void ResetWatchDog();

		void NextStepWatchDog();

   		/*!
   		 * \brief It stops the watchdog.
   		 * 
   		 * It stops the watchdog.
   		 */	
		void StopWatchDog();

   		/*!
   		 * \brief It start the watchdog.
   		 * 
   		 * It start the watchdog.
   		 */	
		void Watchdog();

};

#endif /*REALTIMERESTRICTION_H_*/
