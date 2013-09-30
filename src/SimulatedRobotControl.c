/*******************************************************************************
 *                       SimulatedRobotControl.c                               *
 *                       -----------------------                               *
 * copyright            : (C) 2013 by Richard R. Carrillo and Niceto R. Luque  *
 * email                : rcarrillo, nluque at ugr.es                          *
 *******************************************************************************/

/***************************************************************************
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 ***************************************************************************/

/*!
 * \file SimulatedRobotControl.c
 *
 * \author Niceto R. Luque
 * \author Richard R. Carrillo
 * \date 18 of September 2013
 * In this file the main robot-control loop is implemented.
 */

#if defined(_DEBUG) && (defined(_WIN32) || defined(_WIN64))
#   define _CRTDBG_MAP_ALLOC
#   include <crtdbg.h> // To check for memory leaks
#endif

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

#include <stdio.h>

#include "../include/interface/C_interface_for_robot_control.h"
#include "../include/arm_robot_simulator/ArmRobotSimulation.h"

// Neural-network simulation files
#define NET_FILE "netnew.cfg" // Neural-network definition file used by EDLUT
#define INPUT_WEIGHT_FILE "weights.dat" // Neural-network input weight file used by EDLUT
#define OUTPUT_WEIGHT_FILE "outputweights.dat" // Neural-network output weight file used by EDLUT
#define WEIGHT_SAVE_PERIOD 0.0F // The weights will be saved each period (in seconds) (0=weights are not saved periodically)
#define INPUT_ACTIVITY_FILE NULL // "input.dat" Optional input activity file
#define OUTPUT_ACTIVITY_FILE "output.dat" // Output activity file used to register the neural network activity
#define LOG_FILE "vars.dat"  // Log file used to register the simulation variables
#define REAL_TIME_NEURAL_SIM 0 // EDLUT's simulation mode (0=No real-time neural network simulation 1=For real robot control)

// Robot's dynamics files
#define ROBOT_VAR_FILE_NAME "MANIPULATORS.mat" // MATLAB's file containing the robot's specifications.
#define ROBOT_BASE_VAR_NAME "RRedKuKa" // Variable name in ROBOT_VAR_FILE_NAME which contains the base robot specifications.
#define ROBOT_SIMUL_VAR_NAME "RRedKuKadet10" // Variable name in ROBOT_VAR_FILE_NAME which contains the simulated robot specifications.

#define TRAJ_POS_AMP 0.1 // Amplitude of the desired robot's trajectory
#define TRAJECTORY_TIME 1 // Simulation time in seconds required to execute the desired trajectory once
#define MAX_TRAJ_EXECUTIONS 2 // Maximum number of trajectories repetitions that will be executed by the robot

const double ROBOT_GRAVITY[3]={0, 0, 9.81}; // Earth's standard acceleration due to gravity [Gx Gy Gz]
const double ROBOT_EXTERNAL_FORCE[6]={0, 0, 0, 0, 0, 0}; // External force on manipulator tip [Fx Fy Fz MOMENTUMx MOMENTUMy MOMENTUMz]

///////////////////////////// MAIN LOOP //////////////////////////

int main(int ac, char *av[])
  {
   int errorn;
   long total_output_spks;
   double input_traject_vars[NUM_JOINTS*3]; // Desired trajectory position, velocity and acceletarion
   double robot_state_vars[NUM_JOINTS*3]; // Actual robot position, velocity and acceleration
   double cerebellar_output_vars[NUM_OUTPUT_VARS]={0.0}; // Corrective cerebellar output torque
   double robot_inv_dyn_torque[NUM_JOINTS]; // Robot's inverse dynamics torque
   double total_torque[NUM_JOINTS]; // Total torque applied to the robot
   double robot_error_vars[NUM_JOINTS]; // Joint error (PD correction)
   double cerebellar_learning_vars[NUM_OUTPUT_VARS]; // Error-related learning signals
   // Robot's dynamics variables
   mxArray *robot_inv_dyn_object=NULL, *robot_dir_dyn_object=NULL;
   struct integrator_buffers num_integration_buffers; // Integration buffers used to simulate the robot
   Simulation *neural_sim;
   int n_robot_joints;
   // Time variables
   double sim_time,cur_traject_time;
   float slot_elapsed_time,sim_elapsed_time;
   int n_traj_exec;

   // Variable for logging the simulation state variables
   struct log var_log;

#if defined(_DEBUG) && (defined(_WIN32) || defined(_WIN64))
//   _CrtMemState state0;
   _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
   _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
#endif

#if defined(REAL_TIME_WINNT)
	// Variables for consumed-CPU-time measurement
	LARGE_INTEGER startt,endt,freq;

#elif defined(REAL_TIME_OSX)
	uint64_t startt, endt, elapsed;
	static mach_timebase_info_data_t freq;
#elif defined(REAL_TIME_LINUX)
	// Calculate time taken by a request - Link with real-time library -lrt
	struct timespec startt, endt, freq;
#endif


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

   // Load Matlab robot objects from Matlab files
   if(!(errorn=load_robot(ROBOT_VAR_FILE_NAME,ROBOT_BASE_VAR_NAME,&n_robot_joints,&robot_inv_dyn_object)) && \
      !(errorn=load_robot(ROBOT_VAR_FILE_NAME,ROBOT_SIMUL_VAR_NAME,NULL,&robot_dir_dyn_object)))
     {
      if(!(errorn=allocate_integration_buffers(&num_integration_buffers,n_robot_joints))) // Initialize the buffers for numerical integration using the initial desired robot's state
        {
         // Initialize variable log
         if(!(errorn=create_log(&var_log, MAX_TRAJ_EXECUTIONS, TRAJECTORY_TIME)))
           {
            // Initialize EDLUT and load neural network files
            neural_sim=create_neural_simulation(NET_FILE, INPUT_WEIGHT_FILE, INPUT_ACTIVITY_FILE, OUTPUT_WEIGHT_FILE, OUTPUT_ACTIVITY_FILE, WEIGHT_SAVE_PERIOD, REAL_TIME_NEURAL_SIM);
            if(neural_sim)
              {
               double min_traj_amplitude[3], max_traj_amplitude[3]; // Position, velocity and acceleration
               calculate_input_trajectory_max_amplitude(TRAJECTORY_TIME,TRAJ_POS_AMP, min_traj_amplitude, max_traj_amplitude); // Calcula the maximum and minimum values of the desired trajectory

               total_output_spks=0L;
               puts("Simulating...");
               sim_elapsed_time=0.0;
               errorn=0;
//    _CrtMemCheckpoint(&state0);
               for(n_traj_exec=0;n_traj_exec<MAX_TRAJ_EXECUTIONS && !errorn;n_traj_exec++)
                 {
                  calculate_input_trajectory(robot_state_vars, TRAJ_POS_AMP, 0.0); // Initialize simulated robot's actual state from the desired state (input trajectory) (position, velocity and acceleration)
                  initialize_integration_buffers(robot_state_vars,&num_integration_buffers,n_robot_joints); // For the robot's direct 
                  reset_neural_simulation(neural_sim); // after each trajectory execution the network simulation state must be reset (pending activity events are discarded)
                  cur_traject_time=0.0;
                  do
                    {
                     int n_joint;

#if defined(REAL_TIME_WINNT)
        	QueryPerformanceCounter(&startt);
#elif defined(REAL_TIME_LINUX)
        	clock_gettime(CLOCK_REALTIME, &startt);
#elif defined(REAL_TIME_OSX)
        	startt = mach_absolute_time();
#endif

                     // control loop iteration starts
                     sim_time=(double)n_traj_exec*TRAJECTORY_TIME + cur_traject_time; // Calculate absolute simulation time
                     calculate_input_trajectory(input_traject_vars, TRAJ_POS_AMP, cur_traject_time); // Calculate desired input trajectory
                     //ECEA
                     generate_input_traj_activity(neural_sim, sim_time, input_traject_vars, min_traj_amplitude, max_traj_amplitude); // Translates desired trajectory (position and velocity) into spikes
                     generate_robot_state_activity(neural_sim, sim_time, input_traject_vars, min_traj_amplitude, max_traj_amplitude); // Translates desired trajectory into spikes again to improve the input codification (using the robot's state input neurons)
				     //ICEA
//                   generate_robot_state_activity(neural_sim, sim_time, robot_state_vars, min_traj_amplitude, max_traj_amplitude); // Translates robot's current state (position and velocity) into spikes

                     compute_robot_inv_dynamics(robot_inv_dyn_object,input_traject_vars,ROBOT_EXTERNAL_FORCE,ROBOT_GRAVITY,robot_inv_dyn_torque); // Calculate crude inverse dynamics of the base robot. They constitude the base robot's input torque
                     for(n_joint=0;n_joint<NUM_JOINTS;n_joint++) // Calculate total torque from forward controller (cerebellum) torque plus base controller torque
                        total_torque[n_joint]=robot_inv_dyn_torque[n_joint]+cerebellar_output_vars[n_joint*2]-cerebellar_output_vars[n_joint*2+1];
                     compute_robot_dir_dynamics(robot_dir_dyn_object,robot_state_vars,total_torque,robot_state_vars,&num_integration_buffers,ROBOT_EXTERNAL_FORCE,ROBOT_GRAVITY,SIM_SLOT_LENGTH); // Simulate the robot (direct dynamics).


                     calculate_error_signals(input_traject_vars, robot_state_vars, robot_error_vars); // Calculated robot's performed error
                     calculate_learning_signals(robot_error_vars, cerebellar_output_vars, cerebellar_learning_vars); // Calculate learning signal from the calculated error
                     generate_learning_activity(neural_sim, sim_time, cerebellar_learning_vars); // Translates the learning activity into spikes and injects this activity in the network

                     errorn=run_neural_simulation_slot(neural_sim, sim_time+SIM_SLOT_LENGTH); // Simulation the neural network during a time slot
 
                     total_output_spks+=(long)compute_output_activity(neural_sim, cerebellar_output_vars); // Translates cerebellum output activity into analog output variables (corrective torques)
                     // control loop iteration ends

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
                     sim_elapsed_time+=slot_elapsed_time;
                     log_vars(&var_log, sim_time, input_traject_vars, robot_state_vars, robot_inv_dyn_torque, cerebellar_output_vars, cerebellar_learning_vars, robot_error_vars, slot_elapsed_time,get_neural_simulation_spike_counter(neural_sim)); // Store vars into RAM
                     cur_traject_time+=SIM_SLOT_LENGTH;
                    }
                  while(cur_traject_time<TRAJECTORY_TIME-(SIM_SLOT_LENGTH/2.0) && !errorn); // we add -(SIM_SLOT_LENGTH/2.0) because of floating-point-type codification problems
                 } 
//     reset_neural_simulation(neural_sim);
//     _CrtMemDumpAllObjectsSince(&state0);
               if(errorn)
                  printf("Error %i performing neural network simulation\n",errorn);
               printf("Total neural-network output spikes: %li\n",total_output_spks);
               printf("Total number of neural updates: %Ld\n",get_neural_simulation_event_counter(neural_sim));
               printf("Mean number of neural-network spikes in heap: %f\n",get_accumulated_heap_occupancy_counter(neural_sim)/(double)get_neural_simulation_event_counter(neural_sim));

#if defined(REAL_TIME_WINNT)
               printf("Total elapsed time: %fs (time resolution: %fus)\n",sim_elapsed_time,1.0e6/freq.QuadPart);
#elif defined(REAL_TIME_LINUX)
               printf("Total elapsed time: %fs (time resolution: %fus)\n",sim_elapsed_time,freq.tv_sec*1.0e6+freq.tv_nsec/float(1e3));
#elif defined(REAL_TIME_OSX)
               printf("Total elapsed time: %fs (time resolution: %fus)\n",sim_elapsed_time,1e-3*freq.numer/freq.denom);
#endif

               save_neural_weights(neural_sim);
               finish_neural_simulation(neural_sim);
              }
            else
              {
               errorn=10000;
               printf("Error initializing neural network simulation\n");
              }              
            puts("Saving log file");
            errorn=save_and_finish_log(&var_log, LOG_FILE); // Store logged vars in disk
            if(errorn)
               printf("Error %i while saving log file\n",errorn);
           }
         else
           {
            errorn*=1000;
            printf("Error allocating memory for the log of the simulation variables\n");
           }         
         free_integration_buffers(&num_integration_buffers);
        }
      else
        {
         errorn*=100;
         printf("Error allocating memory for the numerical integration\n");
        }
      free_robot(robot_inv_dyn_object);
      free_robot(robot_dir_dyn_object);
     }
   else
     {
      errorn*=10;
      printf("Error loading the robot object from file: %s\n",ROBOT_VAR_FILE_NAME);
     } 
   if(!errorn)
      puts("OK");
   else
      printf("Error: %i\n",errorn);
#if defined(_DEBUG) && (defined(_WIN32) || defined(_WIN64))
   _CrtDumpMemoryLeaks();
#endif
   return(errorn);
  }
