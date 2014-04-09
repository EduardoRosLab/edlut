/***************************************************************************
 *                  RobotRecursiveInverseDynamics.h					       *
 *                       -------------------	                           *
 * copyright         : (C) 2013 by Richard R. Carrillo and Niceto R. Luque *
 *						and Peter I. Corke								   * 	
 * email             : rcarrillo,nluque at ugr.es                          *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/


#ifndef	_rne_h_
#define	_rne_h_

#include	<math.h>

#include	"MatrixOperations.h"

#define	TRUE	1
#define	FALSE	0

/*!
 * \brief Accessing information within a MATLAB structure is inconvenient and slow. To get around this we build our own robot and link data structures, and copy the information from the MATLAB objects once per call.  If the call is for multiple states values then our efficiency becomes very high.
 */

/* Robot kinematic convention */
typedef
	enum _dhtype {
		STANDARD,
		MODIFIED
} DHType;

/* Link joint type */
typedef
	enum _axistype {
		REVOLUTE = 0,
		PRISMATIC = 1
} Sigma;

/* A robot link structure */
typedef struct _link {
	/**********************************************************
	 *************** kinematic parameters *********************
	 **********************************************************/
	double	alpha;		/* link twist */
	double	A;		/* link offset */
	double	D;		/* link length */
	double	theta;		/* link rotation angle */
	double	offset;		/* link coordinate offset */
	int	sigma; 		/* axis type; revolute or prismatic */

	/**********************************************************
	 ***************** dynamic parameters *********************
	 **********************************************************/

	/**************** of links ********************************/
	Vect	*rbar;		/* centre of mass of link wrt link origin */
	double	m;		/* mass of link */
	double	*I;		/* inertia tensor of link wrt link origin */

	/**************** of actuators *****************************/
		/* these parameters are motor referenced */
	double	Jm;		/* actuator inertia */
	double	G;		/* gear ratio */
	double	B;		/* actuator friction damping coefficient */
	double	*Tc;		/* actuator Coulomb friction coeffient */

	/**********************************************************
	 **************** intermediate variables ******************
	 **********************************************************/
	Vect	r;		/* distance of ith origin from i-1th wrt ith */
	Rot	R;		/* link rotation matrix */
	Vect	omega;		/* angular velocity */
	Vect	omega_d;	/* angular acceleration */
	Vect	acc;		/* acceleration */
	Vect	abar;		/* acceleration of centre of mass */
	Vect	f;		/* inter-link force */
	Vect	n;		/* inter-link moment */
} Link;

/* A robot */
typedef struct _robot {
	int	njoints;	/* number of joints */
	Vect	*gravity;	/* gravity vector */
	DHType	dhtype;		/* kinematic convention */
	Link	*links;		/* the links */
} Robot;

/*!
 * \brief Recursive Newton-Euler algorithm.
 * It computes the inverse dynamics through the recursive Newton-Euler formulation
 * \param robot Pointer to the robot object
 * \param tau Pointer to a vector in which the bias torques will be stored
 * \param qd Pointer to an array containing the current joint velocities
 * \param qdd Pointer to an array containing current joint accelerations
 * \param fext Pointer to an array containing the applied force (or load) at the robot's end
 * \param stride Is used to allow for input and output arrays which are 2-dimensional but in
 *         column-major (Matlab) order. We need to access rows from the arrays.
 */
void newton_euler (Robot *robot, double *tau, double *qd, double *qdd, double *fext, int stride);

#endif
