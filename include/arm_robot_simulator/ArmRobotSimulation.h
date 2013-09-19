/***************************************************************************
 *                      ArmRobotSimulation.h							   *
 *                       -------------------	                           *
 * copyright         : (C) 2013 by Richard R. Carrillo and Niceto R. Luque *
 * email             : rcarrillo, nluque at ugr.es                         *
 ***************************************************************************/

/***************************************************************************
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 ***************************************************************************/

 /*!
 * \file ArmRobotSimulation.h
 *
 * \author Richard R. Carrillo
 * \author Niceto R. Luque
 * \date 19 of September 2013
 *
 * This file declares the functions to simulate the robot's dynamics.
 */

#ifndef _ACCEL_H_
#define _ACCEL_H_

/*!
 * \file accel.h
 *
 * \author Niceto R. Luque
 * \author Richard R. Carrido
 * \date 28 June 2013
 *
 * This file declares all needed functions to calculate the inverse robot dynamics.
 */

#ifndef EXTERN_C
#  ifdef __cplusplus
#    define EXTERN_C extern "C"
#  else
#    define EXTERN_C extern
#  endif
#endif

#include "mex.h"
#include "RobotRecursiveInverseDynamics.h"

/***************************************************************************
 *                                                                         *
 *				DEFINES AND ENUM FOR MANAGING THE ROBOT					   *
 *																		   *		
 *                                                                         *
 ***************************************************************************/

/* Input Arguments */
#define	ROBOT_IN	prhs[0]
#define	A1_IN		prhs[1]
#define	A2_IN		prhs[2]
#define	A3_IN		prhs[3]
#define	A4_IN		prhs[4]
#define	A5_IN		prhs[5]

/* Output Arguments */
#define	TAU_OUT	plhs[0]

/* Some useful things */
#define	NUMROWS(x)	mxGetM(x)
#define	NUMCOLS(x)	mxGetN(x)
#define	NUMELS(x)	(mxGetN(x)*mxGetM(x))
#define	POINTER(x)	mxGetPr(x)

#define INTEGR_BUFFERSIZE 7

/*
 * enums for all the fields we want to pull out of the Matlab robot object
 */
enum {
	FLD_ALPHA,
	FLD_A,
	FLD_THETA,
	FLD_D,
	FLD_SIGMA,
	FLD_OFFSET,
	FLD_M,
	FLD_R,
	FLD_I,
	FLD_JM,
	FLD_G,
	FLD_B,
	FLD_TC,
	FLD_MAX
};
/*
 * struct for all the activity buffers we need to properly integrate acceleration velocity and position.
 */
struct integrator_buffers
  {
   double *velbuffer, *accbuffer;
   double *sumvel, *sumacc;
   double *qinit, *qvinit;
   int occupation;
  };

/* forward defines robot */
static void rot_mat(Link *l, double th, double d, DHType type);
static int mstruct_getfield_number(mxArray *m, char *field);
static int mstruct_getint(mxArray *m, int i, char *field);
static double mstruct_getreal(mxArray *m, int i, char *field);
static double * mstruct_getrealvect(mxArray *m, int i, char *field);
void error(char *s, ...);


/***************************************************************************
 *                                                                         *
 *				FUNTIONS FOR  INTEGRATING ACCELERATION					   *
 *				IN ORDER TO OBTAIN VELOCITY AND POSITION				   *		
 *                                                                         *
 ***************************************************************************/

/*!
 *\brief PURPOSE: Inverse Matrix for calculating the acceleration arguments and sends resultant string to main via and single vector. 
 *
 * Inverting two Matrices needed for obtaining acceleration value of the robot plan when a torque value is applied
 *\param sizeMatrix      size of M1 must be squared
 *\param  M1             PointerEntry Matrix to be multiplied
 *\param  INVM1          Pointer to the result Matrix
 *      
*/
void invermat(int sizeMatrix, double *a, double *ainv);


/* Matrix multiplication functions*/
/*!
 *\brief PURPOSE: Multiplying two Matrices needed for obtaining acceleration value of the robot plan when a torque value is applied
 *\param  M1             Entry Matrix to be multiplied
 *\param  M2             Entry Matrix to be multiplied
 *\param  Result         Pointer to the result Matrix
 *\param  oneRow         number of rows M1
 *\param  oneCol         number of cols M1
 *\param  twoRow         number of rows M2
 *\param  twoCol         number of cols M2
*
*/
void multiplyMatrices(double *M1, double *M2,double *Result,int oneRow, int oneCol, int twoRow, int twoCol);

/*! 
*
*\brief PURPOSE: Integration methods to calcule q and qd
* In order to compute the integral value of the given acceleration and velocity to solve the direct robot dinamic a buffer of 7 elemnts per joint is used to integrate
* the acceleration and velocity. Solving the direct dynamic
*\param		accbuffer     activity acc buffer
*\param		velbuffer     activity velocity buffer
*\param		occupation	  buffer occupation   
*\param		sumacc        accumulated integral
*\param		sumvel        accumulated integral      
*\param	    qddoutput_1	  qdd to be integrated
*\param	    qdoutput	  accumulated integral
*\param     qoutput       accumulated integral
*\param	    qvinit        initial conditions velocity
*\param		qinit         initial conditions possition
*\param	    njoints		  number of robot joints
*\param 	stepsize	  integration step		
*        
*/
void integrationprocedure(double *accbuffer, double *velbuffer,int *occupation,double *sumacc, double *sumvel, double *qddoutput_1 ,double *qdoutput,double *qoutput,double *qvinit, double *qinit,int njoints,double stepsize);

/*!
*\brief Allocates the memory used in the integrative process.
* It allocates the memory of several buffers: sumvel,sumacc,accbuffer,velbuffer,qinit and qvinit.
* For instance, in order to compute the integral value of the given acceleration and velocity to calculate the direct robot 
* dynamics a buffer of 5 elemnts per joint is needed
*\param integr_buffers Pointer to a integrator_buffers structure. This struct will contain the buffer pointers.
*\param njoints Number of robot joints.
*\return The occurred error (0 if the function succeeded).
*
*/
EXTERN_C int allocate_integration_buffers(struct integrator_buffers *integr_buffers, int njoints);

/*!
*
*\brief Frees the memory allocated by allocate_integration_buffers() and used in the integrative process
*
*\param integr_buffers Pointer to a integrator_buffers structure.
*
*/
EXTERN_C void free_integration_buffers(struct integrator_buffers *integr_buffers);

/*!
*
*\brief Initializes the buffers used in the integrative process.
*
*\param entry Pointer to an array containing the initial position, velocity and acceleration per joint (entry values to be accumulated)
*\param integr_buffers Pointer to a integrator_buffers structure to be initialized. This struct must contain allocated buffers.
*\param njoints Number of robot joints.
*
*/
EXTERN_C void initialize_integration_buffers(double *entry, struct integrator_buffers *integr_buffers, int njoints);

/*!
*\brief PURPOSE: approximate the value of a definite integral using boole's rule
*       
*\param   f           buffer containing function points
*\param   h           h(upper limit of integration step-lower limit of integration)        
*\return  y			  approximate value of definite integral
*/
double boolerule ( double *f, double h);

/*!
*\brief PURPOSE: approximate the value of a definite integral using  simsomp's 3/8 rule
*       
*\param   f           buffer containing function points
*\param   h           h(upper limit of integration step-lower limit of integration)        
*\return  y			  approximate value of definite integral
*/
double simp3_8 ( double *f, double h );

/*!
*\brief PURPOSE: approximate the value of a definite integral using  simsomp´s rule
*       
*\param   f           buffer containing function points
*\param   h           h(upper limit of integration step-lower limit of integration)        
*\return  y			  approximate value of definite integral
*/
double simp ( double *f, double h );

/*!
*\brief PURPOSE: approximate the value of a definite integral using the composite trapezoidal rule
*       
*\param   f           buffer containing function points
*\param   h           h(upper limit of integration step-lower limit of integration)        
*\return  y			  approximate value of definite integral
*/
double trap ( double *f, double h );

/*!
*\brief PURPOSE: approximate the value of a definite integral using a fifth order rule
*       
*\param   f           buffer containing function points
*\param   h           h(upper limit of integration step-lower limit of integration)        
*\return  y			  approximate value of definite integral
*/
double fifth ( double *f, double h );

/*!
*\brief PURPOSE: approximate the value of a definite integral using a six order rule
*       
*\param   f           buffer containing function points
*\param   h           h(upper limit of integration step-lower limit of integration)        
*\return  y			  approximate value of definite integral
*/
double sixth ( double *f, double h );


/***************************************************************************
 *                                                                         *
 *				FUNTIONS FOR THE DIREC & INVERSE DYNAMIC   				   *
 *																		   *		
 *                                                                         *
 ***************************************************************************/

/*
*\brief PURPOSE: The plhs[] and prhs[] parameters are vectors that contain pointers to each left-hand side (output) variable and each right-hand side (input) variable, respectively. 
*Accordingly, plhs[0] contains a pointer to the first left-hand side argument, plhs[1] contains a pointer 
*to the second left-hand side argument, and so on. 
*Likewise, prhs[0] contains a pointer to the first right-hand side argument, prhs[1] points to the second, and so on. 
*
*
*\param		nlhs		MATLAB sets nlhs with the number of expected mxArrays.
*\param		plhs		MATLAB sets plhs to a pointer to an array of NULL pointers.
*\param		nrhs		MATLAB sets nrhs to the number of input mxArrays.
*\param		prhs		MATLAB sets prhs to a pointer to an array of input mxArrays. 
*				
*/
void frnecFunction(int	nlhs, mxArray **plhs, int nrhs, const mxArray **prhs);

/*!
*\brief Calculates the robot's inverse dynamics
* This FrnecFuntion Computes inverse dynamics via recursive Newton-Euler formulation
* 
* 	TAU = RNE(ROBOT, Q, QD, QDD)
* 	TAU = RNE(ROBOT, [Q QD QDD])
* 
* 	Returns the joint torque required to achieve the specified joint position,
* 	velocity and acceleration state.
* 
* 	Gravity vector is an attribute of the robot object but this may be 
* 	overriden by providing a gravity acceleration	vector [gx gy gz].
* 
* 	TAU = RNE(ROBOT, Q, QD, QDD, GRAV)
* 	TAU = RNE(ROBOT, [Q QD QDD], GRAV)
* 
* 	An external force/moment acting on the end of the manipulator may also be
* 	specified by a 6-element vector [Fx Fy Fz Mx My Mz].
* 
* 	TAU = RNE(ROBOT, Q, QD, QDD, GRAV, FEXT)
* 	TAU = RNE(ROBOT, [Q QD QDD], GRAV, FEXT)
* 
*	where	Q, QD and QDD are row vectors of the manipulator state; pos, vel, and accel.
* 
*	The torque computed also contains a contribution due to armature
* 	inertia.
*
*	
*\param			 robot				robot variable to be used
*\param			 tray  				position velocity and acceleration values per link from the trajectory generator	
*\param 		 ExternalForce 		external applied Force per link
*\param 		 Gravity			gravity acceleration vector
*\param     	 TorqueOutput		Obtained torque per link
*
*/
EXTERN_C void compute_robot_inv_dynamics(mxArray *robot,double *tray,const double *ExternalForce,const double *Gravity,double *TorqueOutput);

/*!
*\brief PURPOSE:calculate robot's direct dynamics
*        
* Returns a vector of joint accelerations that result from applying the 
* actuator TORQUE to the manipulator ROBOT in state Q and QD.
*
* Uses the method 1 of Walker and Orin to compute the forward dynamics.
* The accelerations of the coordinates are obtained first 
* with the method of Walker-Orin and, later,it is joining to obtain speed and position.  
*
* This form is useful for simulation of manipulator dynamics, in
* conjunction with a numerical integration function.
*
* Walker and Orin is a numerical method used to obtain the acceleration of the
* articular coordinates from the torque vector.For it, Newton-Euler's
* algorithm uses when articular aceleration is zero
* B= 0+H(q,q')+C(q); tau=D(q)q''+B; q''=inv(D(q))[tau-B]
*
*
*    
*\param			  robot						robot variable to be used
*\param			  robot_initial_state		position velocity and acceleration  values per link previously obtained	 
*\param			  external_torque			external applied torque per link
*\param			  robot_resultant_state		Actual position velocity and acceleration  values per link obtained
*\param			  integr_buffers			Activity buffer
*\param			  Gravity					Gravity vector
*\param			  Stepsize					integration step size
*
*/
EXTERN_C void compute_robot_dir_dynamics(mxArray *robot, double *robot_initial_state, double *external_torque, double *robot_resultant_state, struct integrator_buffers *integr_buffers, const double *external_force, const double *gravity, double stepsize);


/***************************************************************************
 *                                                                         *
 *				FUNTIONS FOR CREATING THE TRAJECTORY					   *
 *				AND LOADING THE ROBOT   								   *
 *																		   *		
 *                                                                         *
 ***************************************************************************/

/*Trajectory generator*/

/*!
*\brief PURPOSE: This is for generating an eight like trajectory in cartesian space by means of using sinusoidal curves in joint space
*
*
*
*     
*\param			tsimul	simulation time
*\param         n_joint	number of links the robot has       
*\param     	q          position per link at t= tsimul
*\param 		qd			velocity per link at t=tsimul
*\param 		qdd		acceleration per link at t=tsimul
*
*/
EXTERN_C void trajectory(double *q, double *qd, double *qdd, double tsimul,int n_joints);

/*!
*\brief Loads the robot object from Matlab file
*  in order to dynamically create all the variables needed
*
*\param		  robotfile	   File name from where the robot object is loaded
*\param		  robotname    Robot's variable name in the file
*\param       robot        Pointer to a robot variable pointer which will be set to the loaded robot
*\param		  size         Pointer to the variable where the number of joints will be stored
*\return      occurred error (=0 if no error)
*
*/
EXTERN_C int load_robot(const char *robotfile, const char *robotname, int *size, mxArray **robot);

/*!
*\brief Frees the robot memory array
*
*\param   robot        Robot variable pointer in which the robot variable is allocated
*
*/
EXTERN_C void free_robot(mxArray *robot);


/***************************************************************************
 *                                                                         *
 *				FUNTIONS FOR TESTING DIREC & INVERSE DYNAMIC   			   *
 *																		   *		
 *                                                                         *
 ***************************************************************************/

/*!
*\brief  PURPOSE: This function is for debugging purposes. It simulates the robot during NUMSTEPS*STEPSIZE and creates tra.txt log file
*		  This log file can be plotted with the tra.m Matlab script
*
*/
EXTERN_C void test_simulated_robot_dynamics(void);

#endif /*_ACCEL_H_*/
