/******************************************************************************
 *                      ArmRobotSimulation.c				                  *
 *                      --------------------	                              *
 * copyright            : (C) 2013 by Richard R. Carrillo and Niceto R. Luque *
 * email                : rcarrillo,nluque at ugr.es                          *
 ******************************************************************************/

/***************************************************************************
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 ***************************************************************************/

 /*!
 * \file ArmRobotSimulation.c
 *
 * \author Richard R. Carrillo
 * \author Niceto R. Luque
 * \date 11 of July 2013
 *
 * This file defines the robot dynamics functions.
 */

#include <mex.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <mat.h>
#include <stdlib.h>
#include <time.h>
#include "../../include/arm_robot_simulator/RobotRecursiveInverseDynamics.h"
#include "../../include/arm_robot_simulator/ArmRobotSimulation.h"


/* default values for gravity and external load */
const double GRAVITY[3]={0, 0, 9.81};
const double EXTERNAL_FORCE[6]={0, 0, 0, 0, 0, 0};

/* default values for simulation */
#define NUMSTEPS 500 // Number of integration steps
#define STEPSIZE 0.002 // Integration step size

void test_simulated_robot_dynamics(void)
/*PURPOSE:
           This function is for debugging purposes. It simulates the robot during NUMSTEPS*STEPSIZE and creates tra.txt log file
		   This log file can be plotted with the tra.m Matlab script

     CALLING SEQUENCE:
         test_simulated_robot_dynamics()


     INPUTS:
		  

     OUTPUT:
	 log file tra.txt is created
       	
*/

{
	static int first=0;
	double *qentry,*qdentry,*qddentry;
	double *qtray,*qdtray, *qddtray,*TorqueOutput; 
	double *qddoutput,*qdoutput,*qoutput;	
    struct integrator_buffers robot_integr_buffers;

	double Amplitude=0.1;
	const char *robotfilename = "MANIPULATORS.mat";
	const char *robot = "RRedKuKa"; // "RRedKuKadet10";
	const char *robotDet="RRedKuKa";
    /*Params */
    mxArray *robotdata;
    mxArray *robotdataDet;
    
	int n_joints,i,j;
	FILE *result_trayec;
	double entry[3*3];
	double output[3*3];
	double tray[3*3];


	result_trayec = fopen ("tra.txt","wt");
	if(result_trayec==NULL) {perror("traj file");exit(-1);}
	
	/*	Charging the robot	*/
    if(load_robot(robotfilename, robot, &n_joints, &robotdata)) {perror("robot file");exit(-1);}
    if(load_robot(robotfilename, robotDet, NULL, &robotdataDet)) {perror("robot file");exit(-1);}

    /* Initializing variables */
	qentry=entry;
	qdentry=entry+n_joints;
	qddentry=entry+2*n_joints;
	TorqueOutput = (double*)malloc(n_joints*sizeof(double));

	qtray=tray;
	qdtray=tray+n_joints;
	qddtray=tray+2*n_joints;
	qoutput=output;
	qdoutput=output+n_joints;
	qddoutput=output+2*n_joints;

	for (i=0;i<n_joints;i++){
			qentry[i]=Amplitude*sin((-4*M_PI*0*0*0 + 6*M_PI*0*0) + i*M_PI/4);
	        qdentry[i]=Amplitude*12*M_PI*0*(1 - 0)*cos(4*M_PI*0*0*0 - 6*M_PI*0*0 - M_PI*i/4);
	        qddentry[i]=Amplitude*(12*M_PI*(1-2*0)*cos(4*M_PI*0*0*0 - 6*M_PI*0*0 - M_PI*i/4) + 144*M_PI*M_PI*0*0*(0 - 1)*(0 - 1)*sin(4*M_PI*0*0*0 - 6*M_PI*0*0 - M_PI*i/4));
            
	}
    /*First time loading file and allocating memory*/
		allocate_integration_buffers(&robot_integr_buffers,n_joints);
		initialize_integration_buffers(entry,&robot_integr_buffers,n_joints);

	for(j=1;j<NUMSTEPS;j++){
		/*INVERSE DYNAMICS COMPUTATION*/
		trajectory(qtray,qdtray,qddtray,j*STEPSIZE,n_joints);
		compute_robot_inv_dynamics(robotdata,tray,EXTERNAL_FORCE,GRAVITY,TorqueOutput);

		/*DIRECT DYNAMICS COMPUTATION*/
        compute_robot_dir_dynamics(robotdataDet, entry, TorqueOutput, output, &robot_integr_buffers, EXTERNAL_FORCE, GRAVITY, STEPSIZE);
        memcpy(entry,output,3*n_joints*sizeof(double));
	
        fprintf(result_trayec,"%g ",j*STEPSIZE);
        for(i=0;i<6;i++) {
            const double *log[]={qtray,qdtray,qddtray,qoutput,qdoutput,qddoutput};
            int i2;
            for(i2=0;i2<n_joints;i2++)
		        fprintf(result_trayec,"%g ",log[i][i2]);
		    }
        for(i=0;i<n_joints;i++)
	        fprintf(result_trayec,"%g ",TorqueOutput[i]);
        fprintf(result_trayec,"\n");
		}	

    free_integration_buffers(&robot_integr_buffers);
	free(TorqueOutput);
    free_robot(robotdata);
    fclose(result_trayec);
}


#define NRHS_FASTRNE 6
#define NLHS_FASTRNE 1
void compute_robot_inv_dynamics(mxArray *robot,double *tray,const double *ExternalForce,const double *Gravity,double *TorqueOutput)
/*
PURPOSE:
	FASTRNE	Compute inverse dynamics via recursive Newton-Euler formulation
 
 	TAU = RNE(ROBOT, Q, QD, QDD)
 	TAU = RNE(ROBOT, [Q QD QDD])
 
 	Returns the joint torque required to achieve the specified joint position,
 	velocity and acceleration state.
 
 	Gravity vector is an attribute of the robot object but this may be 
 	overriden by providing a gravity acceleration	vector [gx gy gz].
 
 	TAU = RNE(ROBOT, Q, QD, QDD, GRAV)
 	TAU = RNE(ROBOT, [Q QD QDD], GRAV)
 
 	An external force/moment acting on the end of the manipulator may also be
 	specified by a 6-element vector [Fx Fy Fz Mx My Mz].
 
 	TAU = RNE(ROBOT, Q, QD, QDD, GRAV, FEXT)
 	TAU = RNE(ROBOT, [Q QD QDD], GRAV, FEXT)
 
 	where	Q, QD and QDD are row vectors of the manipulator state; pos, vel, and accel.
 
 	The torque computed also contains a contribution due to armature
 	inertia.

	CALLING SEQUENCE:
      compute_robot_inv_dynamics(mxArray *robot,double *tray,const double *ExternalForce,const double *Gravity,double *TorqueOutput)

     INPUTS:
          robot				robot variable to be used
		  tray  			position velocity and acceleration values per link from the trajectory generator	
		  ExternalForce 	external applied Force per link
		  Gravity			gravity acceleration vector


     OUTPUT:
       	 TorqueOutput		Obtained torque per link


 */

{
int i;
mxArray	*prhs[NRHS_FASTRNE];
mxArray	*plhs[NLHS_FASTRNE];
double *tmp, *tmp2;
int n_rows=1;
int n_cols;
double *qtray, *qdtray, *qddtray;

   /*
	* Creating the entry of the mexfuntion
	*/  	
    n_cols=mstruct_getint(robot, 0, "n");
    qtray=tray;
    qdtray=tray+n_cols;
    qddtray=tray+2*n_cols;
    
    prhs[0]=robot;
	prhs[1] = mxCreateDoubleMatrix(n_rows, n_cols, mxREAL);
	tmp = mxGetPr(prhs[1]);
	/*
	 * position
	 */
	memcpy(tmp, qtray, n_cols*sizeof(double));
	/*
	 * velocity
	 */
	prhs[2] = mxCreateDoubleMatrix(n_rows, n_cols,mxREAL);
	tmp = mxGetPr(prhs[2]);
	memcpy(tmp, qdtray, n_cols*sizeof(double));
	    /*
	 * acceleration
	 */
	prhs[3] = mxCreateDoubleMatrix(n_rows, n_cols,mxREAL);
	tmp = mxGetPr(prhs[3]);
	memcpy(tmp, qddtray, n_cols*sizeof(double));
	/*
	 * gravity[Gx Gy Gz]
	 */
	prhs[4] = mxCreateDoubleMatrix(n_rows,3 ,mxREAL);
	tmp = mxGetPr(prhs[4]);
	memcpy(tmp, Gravity, 3*sizeof(double));	   
	/*Gx*//*Gy//*Gz*/
    /*
	 * External force[Fx Fy Fz Mx My Mz]
	 */
	prhs[5] = mxCreateDoubleMatrix(n_rows,6,mxREAL);
	tmp = mxGetPr(prhs[5]);
	memcpy(tmp, ExternalForce, 6*sizeof(double));
	/*Fx*/ /*Fy*/ /*Fz*/ /*MOMENTUMx*/ /*MOMENTUMy*/ /*MOMENTUMz*/

	frnecFunction(NLHS_FASTRNE,plhs,NRHS_FASTRNE,prhs);
	tmp2 = mxGetPr(plhs[0]);
	memcpy(TorqueOutput,tmp2, n_cols*sizeof(double));
	//for(i=0;i< n_cols;i++)
	//		mexPrintf("Torque Output %lf  \n",tmp2[i]);
			
	for(i=1;i<NRHS_FASTRNE;i++)
       mxDestroyArray(prhs[i]);
	for(i=0;i<NLHS_FASTRNE;i++)
       mxDestroyArray(plhs[i]);
}

/**
 * Computing the direct dynamics
 **/
#define NRHS_ACCEL 5
#define NLHS_ACCEL 1
#define NRHS_ACCEL_TAU 6
#define NLHS_ACCEL_TAU 1
void compute_robot_dir_dynamics(mxArray *robot, double *robot_initial_state, double *external_torque, double *robot_resultant_state, struct integrator_buffers *integr_buffers, const double *external_force, const double *gravity, double stepsize)
/* PURPOSE:
        
 Returns a vector of joint accelerations that result from applying the 
 actuator TORQUE to the manipulator ROBOT in state Q and QD.

 Uses the method 1 of Walker and Orin to compute the forward dynamics.
 The accelerations of the coordinates are obtained first 
 with the method of Walker-Orin and, later,it is joining to obtain speed and position.  

 This form is useful for simulation of manipulator dynamics, in
 conjunction with a numerical integration function.

 Walker and Orin is a numerical method used to obtain the acceleration of the
 articular coordinates from the torque vector.For it, Newton-Euler's
 algorithm uses when articular aceleration is zero
 B= 0+H(q,q')+C(q); tau=D(q)q''+B; q''=inv(D(q))[tau-B]


 CALLING SEQUENCE:
        compute_robot_dir_dynamics(Robot,Robot_initial_state, ExternalTorque,Robot_resultant_state, integr_buffers, External_Force, Gravity, Stepsize)

       

     INPUTS:
          robot						robot variable to be used
		  robot_initial_state		position velocity and acceleration  values per link previously obtained	 
		  external_torque			external applied torque per link
		  robot_resultant_state		Actual position velocity and acceleration  values per link obtained
		  integr_buffers			Activity buffer
		  Gravity					Gravity vector
		  Stepsize					integration step size


     OUTPUT:
		 robot_resultant_state		Actual position velocity and acceleration  values per link obtained

 */
{
static int onetime=0;
int i,k;
mxArray	*prhs[NRHS_ACCEL];
mxArray	*plhs[NLHS_ACCEL];
mxArray	*prhs_tau[NRHS_ACCEL_TAU];
mxArray	*plhs_tau[NLHS_ACCEL_TAU];
double *tmp, *tmp2;
double *Minertia,*InvMinertia, *ExternalTorqueAux;
double *qentrymatrix;
double *qdentrymatrix;
double *qddentrymatrix;
int n_rows=3, n_rows_tau=1;
int n_cols ;
double *qentry, *qdentry, *qddentry;
double *qddoutput, *qdoutput, *qoutput;

/*
****************************************************************************
  compute current manipulator inertia
  torques resulting from unit acceleration of each joint with
  no gravity.
****************************************************************************
*/

   /*
	* Creating the entry of the mexfuntion and initializing Data
	*/
    
    n_cols=mstruct_getint(robot, 0, "n");
    qentry=robot_initial_state;
    qdentry=robot_initial_state+n_cols;
    qddentry=robot_initial_state+2*n_cols;
    qoutput=robot_resultant_state;
    qdoutput=robot_resultant_state+n_cols;
    qddoutput=robot_resultant_state+2*n_cols;

	qentrymatrix = (double*)malloc(n_cols*n_cols*sizeof(double));
	qdentrymatrix = (double*)malloc(n_cols*n_cols*sizeof(double));
	qddentrymatrix = (double*)malloc(n_cols*n_cols*sizeof(double));
	ExternalTorqueAux = (double*)malloc(n_cols*sizeof(double));
	Minertia = (double*)malloc(n_cols*n_cols*sizeof(double));
	InvMinertia=(double*)malloc(n_cols*n_cols*sizeof(double));
	
	/*
	 * position to calculate Matrix Inertia q=[q1 q1 q1;q2 q2 q2;q3 q3 q3]
	 */
	for (i=0;i<n_cols;i++)
		for (k=0;k<n_cols;k++){
				qentrymatrix[i*n_cols+k] = qentry[i];/*position*/
				qdentrymatrix[i*n_cols+k] = 0;/*velocity*/
				if (k==i)
					qddentrymatrix[i*n_cols+k] = 1;
				else
					qddentrymatrix[i*n_cols+k] = 0;

		}
    prhs[0] = robot;
    
	prhs[1] = mxCreateDoubleMatrix(n_cols, n_cols, mxREAL);
	tmp = mxGetPr(prhs[1]);
	memcpy(tmp, qentrymatrix, n_cols*n_cols*sizeof(double));
	/*
	 * velocity to calculate Matrix Inertia qd=[0 0 0;0 0 0;0 0 0]
	 */
	prhs[2] = mxCreateDoubleMatrix(n_cols, n_cols,mxREAL);
	tmp = mxGetPr(prhs[2]);
	memcpy(tmp, qdentrymatrix, n_cols*n_cols*sizeof(double));
    /*
	 * acceleration to calculate Matrix Inertia qdd=[1 0 0;0 1 0;0 0 1]
	 */
	prhs[3] = mxCreateDoubleMatrix(n_cols, n_cols,mxREAL);
	tmp = mxGetPr(prhs[3]);
    memcpy(tmp, qddentrymatrix, n_cols*n_cols*sizeof(double));
	/*
	 * gravity[Gx Gy Gz], to calculate Matrix Inertia G=[0;0;0]
	 */
	prhs[4] = mxCreateDoubleMatrix(n_rows,1 ,mxREAL);
	tmp = mxGetPr(prhs[4]);
		   tmp[0] = 0.0; /*Gx*/
		   tmp[1] = 0.0; /*Gy*/
		   tmp[2] = 0.0; /*Gz*/

	/*
	 * Calculating Matrix Inertia 
	 */

	frnecFunction(NLHS_ACCEL,plhs,NRHS_ACCEL,prhs);
   	tmp2 = mxGetPr(plhs[0]);
	memcpy(Minertia,tmp2,n_cols*n_cols*sizeof(double));
	invermat(n_cols, Minertia, InvMinertia);
	
	for(i=1;i<NRHS_ACCEL;i++)
       mxDestroyArray(prhs[i]);
	for(i=0;i<NLHS_ACCEL;i++)
       mxDestroyArray(plhs[i]);
		
/*
****************************************************************************
	    compute gravity and coriolis torque
	    torques resulting from zero acceleration at given velocity &
	    with gravity acting.
****************************************************************************
*/

	prhs_tau[0] = robot;

	/*
	 * position to calculate coriolis q=[q1 q2 q3]
	 */

	prhs_tau[1] = mxCreateDoubleMatrix(n_rows_tau, n_cols, mxREAL);
	tmp = mxGetPr(prhs_tau[1]);
	memcpy(tmp, qentry, n_cols*sizeof(double));
	
	/*
	 * velocity to calculate coriolis qd=[qd1 qd2 qd3]
	 */
	prhs_tau[2] = mxCreateDoubleMatrix(n_rows_tau, n_cols, mxREAL);
	tmp = mxGetPr(prhs_tau[2]);
	memcpy(tmp, qdentry, n_cols*sizeof(double));
	/*
	 * acceleration to calculate coriolis qd=[0 0 0]
	 */
	prhs_tau[3] = mxCreateDoubleMatrix(n_rows_tau, n_cols, mxREAL);
	tmp = mxGetPr(prhs_tau[3]);
	for(k=0;k<n_cols;k++){
		tmp[k] =0.0;
		}
	/*
	 * gravity[Gx Gy Gz]
	 */
	prhs_tau[4] = mxCreateDoubleMatrix(n_rows_tau, 3, mxREAL);
	tmp = mxGetPr(prhs_tau[4]);
	memcpy(tmp,gravity,sizeof(double)*3);

    /*
	 * External force[Fx Fy Fz Mx My Mz]
	 */
	prhs_tau[5] = mxCreateDoubleMatrix(n_rows_tau, 6, mxREAL);
	tmp = mxGetPr(prhs_tau[5]);
	memcpy(tmp, external_force, 6*sizeof(double));

	frnecFunction(NLHS_ACCEL_TAU,plhs_tau,NRHS_ACCEL_TAU,prhs_tau);
    tmp2 = mxGetPr(plhs_tau[0]);
			for(i=0;i< n_cols;i++)
				ExternalTorqueAux[i]=external_torque[i]-tmp2[i];
			
/*
****************************************************************************
	    compute the acc value within the initial values of qd, qdd
		and torque 
****************************************************************************
*/						

multiplyMatrices(InvMinertia, ExternalTorqueAux,qddoutput,n_rows, n_cols, n_cols, n_rows_tau);

/*
****************************************************************************
	    Integrating the acc value within the initial values of q, qd
		obtaining q and qd of the robot plant 
****************************************************************************
*/
integrationprocedure(integr_buffers->accbuffer, integr_buffers->velbuffer, &integr_buffers->occupation, integr_buffers->sumacc, integr_buffers->sumvel, qddoutput,qdoutput,qoutput,integr_buffers->qvinit,integr_buffers->qinit,n_cols,stepsize);


	for(i=1;i<NRHS_ACCEL_TAU;i++)
       mxDestroyArray(prhs_tau[i]);
	for(i=0;i<NLHS_ACCEL_TAU;i++)
       mxDestroyArray(plhs_tau[i]);

free(qentrymatrix);
free(qdentrymatrix);
free(qddentrymatrix);
free(InvMinertia);
free(Minertia);
free(ExternalTorqueAux);
}
/**
*  RNE can be either an M-file or a MEX-file.  This is the Mexfile for C
*  See the robotic toolbox manual for details on how to configure the MEX-file.  
*  The M-file is a wrapper which calls either
*  RNE_DH or RNE_MDH depending on the kinematic conventions used by the robot
*  object.
*
*
**/

/**
 * Mex function entry point.
 */
void 
frnecFunction(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
/*
PURPOSE:
The plhs[] and prhs[] parameters are vectors that contain pointers to each left-hand side (output) 
variable and each right-hand side (input) variable, respectively. 
Accordingly, plhs[0] contains a pointer to the first left-hand side argument, plhs[1] contains a pointer 
to the second left-hand side argument, and so on. 
Likewise, prhs[0] contains a pointer to the first right-hand side argument, prhs[1] points to the second, and so on. 

  CALLING SEQUENCE:
	Arguments
		nlhs		MATLAB sets nlhs with the number of expected mxArrays.
		plhs		MATLAB sets plhs to a pointer to an array of NULL pointers.
		nrhs		MATLAB sets nrhs to the number of input mxArrays.
		prhs		MATLAB sets prhs to a pointer to an array of input mxArrays. 
					
	Description
		These mxArrays are declared as constant, 
		they are read only and should not be modified by your MEX-file. 
		Changing the data in these mxArrays may produce undesired side effects.
		mexFunction is not a routine you call. Rather, mexFunction is the generic name of the function 
		entry point that must exist in every C source MEX-file. 
		When you invoke a MEX-function, MATLAB finds and loads the corresponding MEX-file of the same name.
		MATLAB then searches for a symbol named mexFunction within the MEX-file. If it finds one, 
		it calls the MEX-function using the address of the mexFunction symbol. 
		If MATLAB cannot find a routine named mexFunction inside the MEX-file, it issues an error message.
		When you invoke a MEX-file, MATLAB automatically seeds nlhs, plhs, nrhs, and prhs with the caller's information. 
		In the syntax of the MATLAB language, functions have the general form

		[a,b,c,...] = fun(d,e,f,...)

		where the denotes more items of the same format. 
		The a,b,c... are left-hand side arguments and the 
		d,e,f... are right-hand side arguments. 
		The arguments nlhs and nrhs contain the number of left-hand side and right-hand side arguments, 
		respectively, with which the MEX-function is called. prhs is a pointer to a length nrhs array 
		of pointers to the right-hand side mxArrays. plhs is a pointer to a length nlhs array where your
		C function must put pointers for the returned left-hand side mxArrays.

*/
{
	double	*q, *qd, *qdd;
	double	*tau;
//	unsigned int	m,n;
	int	j, njoints, p, nq; // l
	double	*fext = NULL;
	double *grav = NULL;
	Robot		robot;
	mxArray		*link0;
	mxArray		*mx_robot;
	mxArray		*mx_links;
	static int	firstime = 0;
	int	fieldmap[FLD_MAX];

	if (  !mxIsStruct(ROBOT_IN) ||
	      (strcmp(mxGetClassName(ROBOT_IN), "robot") != 0)
	)
		mexErrMsgTxt("first argument is not a robot structure\n");


	mx_robot = (mxArray *)ROBOT_IN;
	njoints = mstruct_getint(mx_robot, 0, "n");

/*
**********************************************************************
 * Handle the different calling formats.
 * Setup pointers to q, qd and qdd inputs 
 **********************************************************************
 */

	switch (nrhs) {
	case 2:
	/*
	 * TAU = RNE(ROBOT, [Q QD QDD])
	 */ 
		if (NUMCOLS(A1_IN) != 3 * njoints)
			mexErrMsgTxt("RNE too few cols in [Q QD QDD]");
		q = POINTER(A1_IN);
		nq = NUMROWS(A1_IN);
		qd = &q[njoints*nq];
		qdd = &q[2*njoints*nq];
		break;
		
	case 3:
	/*
	 * TAU = RNE(ROBOT, [Q QD QDD], GRAV)
	 */ 
		if (NUMCOLS(A1_IN) != (3 * njoints))
			mexErrMsgTxt("RNE too few cols in [Q QD QDD]");
		q = POINTER(A1_IN);
		nq = NUMROWS(A1_IN);
		qd = &q[njoints*nq];
		qdd = &q[2*njoints*nq];

		if (NUMELS(A2_IN) != 3)
			mexErrMsgTxt("RNE gravity vector expected");
		grav = POINTER(A2_IN);
		break;

	case 4:
	/*
	 * TAU = RNE(ROBOT, Q, QD, QDD)
	 * TAU = RNE(ROBOT, [Q QD QDD], GRAV, FEXT)
	 */ 
		if (NUMCOLS(A1_IN) == (3 * njoints)) {
			q = POINTER(A1_IN);
			nq = NUMROWS(A1_IN);
			qd = &q[njoints*nq];
			qdd = &q[2*njoints*nq];

			if (NUMELS(A2_IN) != 3)
				mexErrMsgTxt("RNE gravity vector expected");
			grav = POINTER(A2_IN);
			if (NUMELS(A3_IN) != 6)
				mexErrMsgTxt("RNE Fext vector expected");
			fext = POINTER(A3_IN);
		} else {
			int	nqd = NUMROWS(A2_IN),
				nqdd = NUMROWS(A3_IN);

			nq = NUMROWS(A1_IN);
			if ((nq != nqd) || (nqd != nqdd))
				mexErrMsgTxt("RNE Q QD QDD must be same length");
			if ( (NUMCOLS(A1_IN) != njoints) ||
			     (NUMCOLS(A2_IN) != njoints) ||
			     (NUMCOLS(A3_IN) != njoints)
			) 
				mexErrMsgTxt("RNE Q must have Naxis columns");
			q = POINTER(A1_IN);
			qd = POINTER(A2_IN);
			qdd = POINTER(A3_IN);
		}
		break;

	case 5: {
	/*
	 * TAU = RNE(ROBOT, Q, QD, QDD, GRAV)
	 */
		int	nqd = NUMROWS(A2_IN),
			nqdd = NUMROWS(A3_IN);

		nq = NUMROWS(A1_IN);
		if ((nq != nqd) || (nqd != nqdd))
			mexErrMsgTxt("RNE Q QD QDD must be same length");
		if ( (NUMCOLS(A1_IN) != njoints) ||
		     (NUMCOLS(A2_IN) != njoints) ||
		     (NUMCOLS(A3_IN) != njoints)
		) 
			mexErrMsgTxt("RNE Q must have Naxis columns");
		q = POINTER(A1_IN);
		qd = POINTER(A2_IN);
		qdd = POINTER(A3_IN);
		if (NUMELS(A4_IN) != 3)
			mexErrMsgTxt("RNE gravity vector expected");
		grav = POINTER(A4_IN);
		break;
	}

	case 6: {
	/*
	 * TAU = RNE(ROBOT, Q, QD, QDD, GRAV, FEXT)
	 */
		int	nqd = NUMROWS(A2_IN),
			nqdd = NUMROWS(A3_IN);

		nq = NUMROWS(A1_IN);
		if ((nq != nqd) || (nqd != nqdd))
			mexErrMsgTxt("RNE Q QD QDD must be same length");
		if ( (NUMCOLS(A1_IN) != njoints) ||
		     (NUMCOLS(A2_IN) != njoints) ||
		     (NUMCOLS(A3_IN) != njoints)
		) 
			mexErrMsgTxt("RNE Q must have Naxis columns");
		q = POINTER(A1_IN);
		qd = POINTER(A2_IN);
		qdd = POINTER(A3_IN);
		if (NUMELS(A4_IN) != 3)
			mexErrMsgTxt("RNE gravity vector expected");
		grav = POINTER(A4_IN);
		if (NUMELS(A5_IN) != 6)
			mexErrMsgTxt("RNE Fext vector expected");
		fext = POINTER(A5_IN);
		break;
	}
	default:
		mexErrMsgTxt("RNE wrong number of arguments.");
	}


	/*
	 * fill out the robot structure
	 */
	robot.njoints = njoints;

	if (grav)
		robot.gravity = (Vect *)grav;
	else
		robot.gravity = (Vect *)mxGetPr( mxGetField(mx_robot, 0, "gravity") );
	robot.dhtype = mstruct_getint(mx_robot, 0, "mdh");

	/* build link structure */
	robot.links = (Link *)mxCalloc(njoints, sizeof(Link));


/*
**********************************************************************
 * Now we have to get pointers to data spread all across a cell-array
 * of Matlab structures.
 *
 * Matlab structure elements can be found by name (slow) or by number (fast).
 * We assume that since the link structures are all created by the same
 * constructor, the index number for each element will be the same for all
 * links.  However we make no assumption about the numbers themselves.
 **********************************************************************
 */

	/* get pointer to link structures */
	mx_links = mxGetField(mx_robot, 0, "link");
	if (mx_links == NULL)
		mexErrMsgTxt("couldnt find element link in robot structure");

	/*
	 * Elements of the link structure are:
	 *
	 *	alpha: 
	 *	A:
	 *	theta:
	 *	D:
	 *	offset:
	 *	sigma:
	 *	mdh:
	 *	m:
	 *	r:
	 *	I:
	 *	Jm:
	 *	G:
	 *	B:
	 *	Tc:
	 */

	/* take the first link as the template */
	link0 = mxGetCell(mx_links, 0);

	/* and lookup each structure element, and save the index */
	fieldmap[FLD_ALPHA] = mstruct_getfield_number(link0, "alpha");
	fieldmap[FLD_A] = mstruct_getfield_number(link0, "A");
	fieldmap[FLD_THETA] = mstruct_getfield_number(link0, "theta");
	fieldmap[FLD_D] = mstruct_getfield_number(link0, "D");
	fieldmap[FLD_SIGMA] = mstruct_getfield_number(link0, "sigma");
	fieldmap[FLD_OFFSET] = mstruct_getfield_number(link0, "offset");
	fieldmap[FLD_M] = mstruct_getfield_number(link0, "m");
	fieldmap[FLD_R] = mstruct_getfield_number(link0, "r");
	fieldmap[FLD_I] = mstruct_getfield_number(link0, "I");
	fieldmap[FLD_JM] = mstruct_getfield_number(link0, "Jm");
	fieldmap[FLD_G] = mstruct_getfield_number(link0, "G");
	fieldmap[FLD_B] = mstruct_getfield_number(link0, "B");
	fieldmap[FLD_TC] = mstruct_getfield_number(link0, "Tc");

	/*
	 * now for each link structure, use the saved index to copy the
	 * data into the corresponding Link structure.
	 */
	for (j=0; j<njoints; j++) {
		Link	*l = &robot.links[j];

		mxArray	*m = mxGetCell(mx_links, j);

		l->alpha = mxGetScalar( mxGetFieldByNumber(m, 0, fieldmap[FLD_ALPHA]) );
		l->A = mxGetScalar( mxGetFieldByNumber(m, 0, fieldmap[FLD_A]) );
		l->theta = mxGetScalar( mxGetFieldByNumber(m, 0, fieldmap[FLD_THETA]) );
		l->D = mxGetScalar( mxGetFieldByNumber(m, 0, fieldmap[FLD_D]) );
		l->sigma = (int)mxGetScalar( mxGetFieldByNumber(m, 0, fieldmap[FLD_SIGMA]) );
		l->offset = mxGetScalar( mxGetFieldByNumber(m, 0, fieldmap[FLD_OFFSET]) );
		l->m = mxGetScalar( mxGetFieldByNumber(m, 0, fieldmap[FLD_M]) );
		l->rbar = (Vect *)mxGetPr( mxGetFieldByNumber(m, 0, fieldmap[FLD_R]) );
		l->I = mxGetPr( mxGetFieldByNumber(m, 0, fieldmap[FLD_I]) );
		l->Jm = mxGetScalar( mxGetFieldByNumber(m, 0, fieldmap[FLD_JM]) );
		l->G = mxGetScalar( mxGetFieldByNumber(m, 0, fieldmap[FLD_G]) );
		l->B = mxGetScalar( mxGetFieldByNumber(m, 0, fieldmap[FLD_B]) );
		l->Tc = mxGetPr( mxGetFieldByNumber(m, 0, fieldmap[FLD_TC]) );
	
		
	}
	/* Create a matrix for the return argument */
	TAU_OUT = mxCreateDoubleMatrix(nq, njoints, mxREAL);
	tau = mxGetPr(TAU_OUT);

#define	MEL(x,R,C)	(x[(R)+(C)*nq])

	/* for each point in the input trajectory */
	for (p=0; p<nq; p++) {
		/*
		 * update all position dependent variables
		 */
		for (j = 0; j < njoints; j++) {
			Link	*l = &robot.links[j];

			switch (l->sigma) {
			case REVOLUTE:
				rot_mat(l, MEL(q,p,j)+l->offset, l->D, robot.dhtype);
				break;
			case PRISMATIC:
				rot_mat(l, l->theta, MEL(q,p,j)+l->offset, robot.dhtype);
				break;
			}
#ifdef	DEBUG
			rot_print("R", &l->R);
			vect_print("p*", &l->r);
#endif
		}

		newton_euler(&robot, &tau[p], &qd[p], &qdd[p], fext, nq);


		

	}

	mxFree(robot.links);
}

/*
 *	Written by;
 *
 *		Peter I. Corke
 *		CSIRO Division of Manufacturing Technology
 *		Preston, Melbourne.  Australia. 3072.
 *		pic@mlb.dmt.csiro.au
 *		Niceto R. Luque Sola
 *		Univerity of Granada. Spain
 *		nluque@ugr.es
 *
 *
 *
 *
 */

/**
 * Return the link rotation matrix and translation vector.
 *
 * @param l Link object for which R and p* are required.
 * @param th Joint angle, overrides value in link object
 * @param d Link extension, overrides value in link object
 * @param type Kinematic convention.
 */
static void
rot_mat (
	Link	*l,
	double	th,
	double	d,
	DHType	type
) {
	double		st, ct, sa, ca;

#ifdef	sun
	sincos(th, &st, &ct);
	sincos(l->alpha, &sa, &ca);
#else
	st = sin(th);
	ct = cos(th);
	sa = sin(l->alpha);
	ca = cos(l->alpha);
#endif

	switch (type) {
case STANDARD:
	l->R.n.x = ct;		l->R.o.x = -ca*st;	l->R.a.x = sa*st;
	l->R.n.y = st;		l->R.o.y = ca*ct;	l->R.a.y = -sa*ct;
	l->R.n.z = 0.0;		l->R.o.z = sa;		l->R.a.z = ca;

	l->r.x = l->A;
	l->r.y = d * sa;
	l->r.z = d * ca;
	break;
case MODIFIED:
	l->R.n.x = ct;		l->R.o.x = -st;		l->R.a.x = 0.0;
	l->R.n.y = st*ca;	l->R.o.y = ca*ct;	l->R.a.y = -sa;
	l->R.n.z = st*sa;	l->R.o.z = ct*sa;	l->R.a.z = ca;

	l->r.x = l->A;
	l->r.y = -d * sa;
	l->r.z = d * ca;
	break;
	}
}

/*************************************************************************
 * Matlab structure access methods
 *************************************************************************/
static mxArray *
mstruct_get_element(mxArray *m, int i, char *field)
{
	mxArray	*e;

	if (mxIsCell(m)) {
#ifdef	DEBUG
		mexPrintf("%d x %d\n", mxGetM(m), mxGetN(m));
#endif
		/* get the i'th cell from the cell array */
		if ((e = mxGetCell(m, i)) == NULL)
			error("get_element: field %s: cant get cell element %d",field, i);
	} else
		e = m;

	if (!mxIsStruct(e))
		mexErrMsgTxt("get_element: expecting a structure");
	if ((e = mxGetField(e, 0, field)) != NULL)
		return e;
	else {
		error("No such field as %s", field);
		return NULL;
	}
}

static int
mstruct_getfield_number(mxArray *m, char *field)
{
	int	f;
	
	if ((f = mxGetFieldNumber(m, field)) < 0)
		error("no element %s in link structure");

	return f;
}

static int
mstruct_getint(mxArray *m, int i, char *field)
{
	mxArray	*e;

	e = mstruct_get_element(m, i, field);

	return (int) mxGetScalar(e);
}

static double
mstruct_getreal(mxArray *m, int i, char *field)
{
	mxArray	*e;

	e = mstruct_get_element(m, i, field);

	return mxGetScalar(e);
}

static double *
mstruct_getrealvect(mxArray *m, int i, char *field)
{
	mxArray	*e;

	e = mstruct_get_element(m, i, field);

	return mxGetPr(e);
}

#include	<stdarg.h>

/**
 * Error message handler.  Takes printf() style format string and variable
 * arguments and sends resultant string to Matlab via \t mexErrMsgTxt().
 *
 * @param s Error message string, \t  printf() style.
 */
void
error(char *s, ...)
{
	char	b[BUFSIZ];

	va_list	ap;

	va_start(ap, s);

	vsprintf(b, s, ap);

	mexErrMsgTxt(b);
}
/**
 * Inverse Matrix for calculating the acceleration
 * arguments and sends resultant string to main via and single vector.
 * entries: size of the matrix, the original matrix and the inverse matrix
 * */

void
invermat(int sizeMatrix, double *a, double *ainv) 
/*PURPOSE:
         Inverting two Matrices needed for obtaining acceleration value of the robot plan
         when a torque value is applied

     CALLING SEQUENCE:
         invermat(sizeMatrix, M1,INVM1)  

     INPUTS:
		  sizeMatrix     size of M1 must be squared
          M1             PointerEntry Matrix to be multiplied
          INVM1          Pointer to the result Matrix
         

     OUTPUT:
          INVM1          Inverse Matrix

*/
{
	double coef, value,element;
	double *aux= (double*)malloc(sizeMatrix*sizeMatrix*sizeof(double));
	int i,j,n,m,s;
	for (i=0;i<sizeMatrix;i++){
		for(j=0;j<sizeMatrix;j++){
			if(i==j)
				ainv[i*sizeMatrix+j]=1.0;
			else
				ainv[i*sizeMatrix+j]=0.0;
		}
	}

	/*Iterations*/
	for (s=0;s<sizeMatrix;s++){
		element=a[s*sizeMatrix+s];
		if(element==0){
			for(n=s+1; n<sizeMatrix; n++){
				element=a[n*sizeMatrix+s];
				if(element!=0){
					for(m=0; m<sizeMatrix; m++){
						value=a[n*sizeMatrix+m];
						a[n*sizeMatrix+m]=a[s*sizeMatrix+m];
						a[s*sizeMatrix+m]=value;
					}
					break;
				}
				if(n==(sizeMatrix-1)){
					printf("This matrix is not invertible\n");
					exit(0);
				}
			}
		}
		for (j=0;j<sizeMatrix;j++){
			a[s*sizeMatrix+j]/=element;
			ainv[s*sizeMatrix+j]/=element;
		}
	
		for(i=0;i<sizeMatrix;i++)
		{
			if (i!=s){
				coef=a[i*sizeMatrix+s];
				for (j=0;j<sizeMatrix;j++){
					aux[j]=a[s*sizeMatrix+j]*(coef*-1);
					aux[sizeMatrix+j]=ainv[s*sizeMatrix+j]*(coef*-1);
				}
				for (j=0;j<sizeMatrix;j++){
					a[i*sizeMatrix+j]+=aux[j];
					ainv[i*sizeMatrix+j]+=aux[sizeMatrix+j];
				}
			}
		}
	}
	free(aux);
return;
}

/**
 *
 * This is for multiplying Matrix. 
 *
 * */

void multiplyMatrices(double *M1, double *M2,double *Result,int oneRow, int oneCol, int twoRow, int twoCol)
/*
     PURPOSE:
         Multiplying two Matrices needed for obtaining acceleration value of the robot plan
         when a torque value is applied

     CALLING SEQUENCE:
          multiplyingMatrices( M1,M2, &Result, oneRow, oneCol, twoRow, twoCol) 

     INPUTS:
          M1             Entry Matrix to be multiplied
          M2             Entry Matrix to be multiplied
          Result         Pointer to the result Matrix
          oneRow         number of rows M1
		  oneCol         number of cols M1
		  twoRow         number of rows M2
		  twoCol         number of cols M2


     OUTPUT:
          Result        result Matrix

*/
/*Start of multiplyMatrices function*/
{
/*Declare variables*/
int  i, j, k;
/*Create the arrays in the heap memory*/


/*The number of columns in oneMatrix must equal the number of rows in twoMatrix*/
if(oneCol != twoRow)
{
printf("The number of columns in the first matrix must equal the number of rows in the second matrix.\n");
exit(EXIT_FAILURE);
}

/*Declare pointer variables to store the initial locations of my arrays for later use
 for loop while i is less than oneRow(number of rows in first matrix)*/
for(i = 0; i < oneRow; i++)
{	/*A for loop while j is less than twoCol*/
	for(j = 0; j < twoCol; j++)
	{
	/* for loop while k is less than oneCol*/
		Result[i*twoCol+j]=0.0;
		for(k = 0; k < oneCol; k++)
		{
	/*Multiply the current entries of the matrices and add the answer to the current entry for the multiplied matrix*/
			Result[ i*twoCol+j ] += M1[ i*oneCol+ k ]*M2[ k*twoCol + j ] ;
			
			
		}
	}
}


return;

}

/**
 *
 * This is the integration method for approximating the value of acc 
 *         
 *
 * */


double trap ( double *f, double h )

/*
     PURPOSE:
          approximate the value of a definite integral using
          the composite trapezoidal rule

     CALLING SEQUENCE:
          y = trap ( *f, h);
          

     INPUTS:
          f             buffer containing function points
          h             h(upper limit of integration step-lower limit of integration)
          


     OUTPUT:
          y		approximate value of definite integral
*/

{
     double sum;

     
     sum = 0.5 * ( f[0] + f[1] );
     return ( h*sum );
}



double simp ( double *f, double h )

/*
     PURPOSE:
          approximate the value of a definite integral using
          the composite Simpson's rule

     CALLING SEQUENCE:
          y = simp ( *f, h);
         

     INPUTS:
          f             buffer containing function points
          h             h(upper limit of integration step-lower limit of integration)
         


     OUTPUT:
          y		approximate value of definite integral
*/

{     
     double haux,
            sum,m;

     haux = 2.0*h;
     m = haux /6.0;
     sum =m* (f[0] + 4.0*f[1]+ f[2]);
     
         
     return ( sum );
}

double simp3_8 ( double *f, double h )

/*
     PURPOSE:
          approximate the value of a definite integral using
          the composite 3/8 Simpson's rule

     CALLING SEQUENCE:
          y = simp ( *f, h);
          

     INPUTS:
          f             buffer containing function points
          h             h(upper limit of integration step-lower limit of integration)
          
     OUTPUT:
          y		approximate value of definite integral
*/

{     
     double haux,
            sum,m;

     haux = 3.0*h;
     m = haux /8.0;
     sum =m* (f[0] + 3.0*f[1]+ 3.0*f[2]+ f[3]);
             
     return ( sum );
}
double boolerule ( double *f, double h)

/*
     PURPOSE:
          approximate the value of a definite integral using
          the composite boolerule

     CALLING SEQUENCE:
          y = bool ( *f, h );
          
     INPUTS:
          f             buffer containing function points
          h             h(upper limit of integration step-lower limit of integration)
         

     OUTPUT:
          y		approximate value of definite integral
*/

{     
     double haux,
            sum,m;

     haux = 4.0*h;
     m = haux /90.0;
     sum =m* (7.0*f[0] + 32.0*f[1]+ 12.0*f[2]+ 32.0*f[3]+7.0*f[4]);
             
     return ( sum );
}

double fifth ( double *f, double h )
/*
     PURPOSE:
          approximate the value of a definite integral using
          the composite boolerule

     CALLING SEQUENCE:
          y = fifth ( *f, h);
         
     INPUTS:
          f             buffer containing function points
          h             h(upper limit of integration step-lower limit of integration)
          
     OUTPUT:
          y		approximate value of definite integral
*/

{     
     double haux,
            sum,m;

     haux = 5.0*h;
     m = haux /288.0;
     sum =m* (19.0*f[0] + 75.0*f[1]+ 50.0*f[2]+ 50.0*f[3]+75.0*f[4]+19.0*f[5]);
             
     return ( sum );
}
double sixth ( double *f, double h )
/*
     PURPOSE:
          approximate the value of a definite integral using
          the composite boolerule

     CALLING SEQUENCE:
          y = sixth ( *f, h);
          
     INPUTS:
          f             buffer containing function points
          h             h(upper limit of integration step-lower limit of integration)
          

     OUTPUT:
          y		approximate value of definite integral
*/

{     
     double haux,
            sum,m;

     haux = 6.0*h;
     m = haux /840.0;
     sum =m* (41.0*f[0] + 216.0*f[1]+ 27.0*f[2]+ 272.0*f[3]+27.0*f[4]+216.0*f[5]+41.0*f[6]);
             
     return ( sum );
}

int allocate_integration_buffers(struct integrator_buffers *integr_buffers, int njoints)
/*  PURPOSE:
         This is to allocate the memory used in the integrative process

     CALLING SEQUENCE:
         allocate_integration_buffers(integr_buffers) 

     INPUTS:
		  integr_buffers   allocate buffer memory sumvel,sumacc,accbuffer,velbuffer,qinit and qvinit 
	 OUTPUT:
		  return 1 if error

*/
  {
   int ret;
   integr_buffers->sumvel=(double*)malloc(njoints*sizeof(double));
   integr_buffers->sumacc=(double*)malloc(njoints*sizeof(double));
   integr_buffers->accbuffer=(double*)malloc(INTEGR_BUFFERSIZE*njoints*sizeof(double));
   integr_buffers->velbuffer=(double*)malloc(INTEGR_BUFFERSIZE*njoints*sizeof(double));
   integr_buffers->qinit=(double*)malloc(njoints*sizeof(double));
   integr_buffers->qvinit=(double*)malloc(njoints*sizeof(double));
   if(integr_buffers->sumvel==NULL || integr_buffers->sumacc==NULL || integr_buffers->accbuffer==NULL || integr_buffers->velbuffer==NULL || integr_buffers->qinit==NULL || integr_buffers->qvinit==NULL)
      {
       perror("Allocating memory for the buffers");
	   free(integr_buffers->sumvel);
	   free(integr_buffers->sumacc);
	   free(integr_buffers->accbuffer);
	   free(integr_buffers->velbuffer);
	   free(integr_buffers->qinit);
	   free(integr_buffers->qvinit);
	   ret=1;
	  }
   else
      ret=0;
   return(ret);
  }
  
/**
 * Initializing buffer accel and velocity
 * 
 **/


void initialize_integration_buffers(double *entry, struct integrator_buffers *integr_buffers, int njoints)
/*PURPOSE:
         In order to compute the integral value of the given acceleration and velocity to
         solve the direct robot dinamic a buffer of 5 elemnts per joint is needed. This is for initializing 
		 these buffers

     CALLING SEQUENCE:
        initialize_integration_buffers(entry,*integr_buffers,njoints) 

     INPUTS:
		  entry:				initial entry values to be accumulated
		  integrator_buffers:	struct with the buffer to be initialized
		  njoints:				number of joints	
         

     OUTPUT:
                                struct with the initialized buffers

*/
{
	int i,j;
    double *qentry, *qdentry, *qddentry;
    qentry=entry;
    qdentry=entry+njoints;
    qddentry=entry+2*njoints;
    for (i=0;i<njoints;i++){
        (integr_buffers->sumvel)[i]=0.0;
        (integr_buffers->sumacc)[i]=0.0;
        (integr_buffers->qinit)[i]=qentry[i];
        (integr_buffers->qvinit)[i]=qdentry[i];
	    for(j=0;j<INTEGR_BUFFERSIZE;j++){
   			if (j==0){
		        (integr_buffers->accbuffer)[i*INTEGR_BUFFERSIZE+j]=qddentry[i];
   			    (integr_buffers->velbuffer)[i*INTEGR_BUFFERSIZE+j]=qdentry[i];
		    }
		    else{
    		    (integr_buffers->accbuffer)[i*INTEGR_BUFFERSIZE+j]=0.0;
	    	    (integr_buffers->velbuffer)[i*INTEGR_BUFFERSIZE+j]=0.0;
	    	}
	    }
	}
    integr_buffers->occupation=1;
}

/**
 * FREE THE MEMORY
 * */
void free_integration_buffers(struct integrator_buffers *integr_buffers)
/*PURPOSE:
         This is to free the memory used in the integrative process

     CALLING SEQUENCE:
         free_integration_buffers(integr_buffers) 

     INPUTS:
		  integr_buffers   free activity sumvel,sumacc,accbuffer,velbuffer,qinit and qvinit 
*/
{
    free(integr_buffers->sumvel);
    free(integr_buffers->sumacc);
    free(integr_buffers->accbuffer);
    free(integr_buffers->velbuffer);
    free(integr_buffers->qinit);
    free(integr_buffers->qvinit);
}
/**
 * managing the activity buffer accel and velocity,making the integrative process
 * */
void integrationprocedure(double *accbuffer, double *velbuffer,int *buff_occup,double *sumacc, double *sumvel, double *qddoutput_1 ,double *qdoutput,double *qoutput,double *qvinit, double *qinit,int njoints,double stepsize)
 /*PURPOSE:
         In order to compute the integral value of the given acceleration and velocity to
         solve the direct robot dinamic a buffer of 7 elemnts per joint is used to integrate
		 the acceleration and velocity. Solving the direct dynamic

     CALLING SEQUENCE:
         integrationprocedure(accbuffer, velbuffer,buff_occup,&sumacc, &sumvel, qddoutput_1,qdoutput,qoutput,qvinit,qinit,njoints,stepsize) 


     INPUTS:
		accbuffer       activity acc buffer
		velbuffer       activity velocity buffer
		occupation	    buffer occupation   
		sumacc          accumulated integral
		sumvel          accumulated integral      
		qddoutput_1	    qdd to be integrated
		qdoutput	    accumulated integral
		qoutput         accumulated integral
		qvinit          initial conditions velocity
		qinit           initial conditions possition
		njoints			number of robot joints
		stepsize		integration step		
         

     OUTPUT:
       	 qdoutput            accumulated integral
         qoutput             accumulated integral

*/
{
int i;


switch(*buff_occup) {

	case 1:
		for (i=0;i<njoints;i++){
				 accbuffer[i*INTEGR_BUFFERSIZE+*buff_occup]=qddoutput_1[i];
				 qdoutput[i]=trap(accbuffer+i*INTEGR_BUFFERSIZE, stepsize)+qvinit[i];
				 velbuffer[i*INTEGR_BUFFERSIZE+*buff_occup]=qdoutput[i];
			     qoutput[i]=trap(velbuffer+i*INTEGR_BUFFERSIZE, stepsize)+qinit[i];
				 
		}
				
		(*buff_occup)++;
		break;
	case 2:
		for (i=0;i<njoints;i++){
				accbuffer[i*INTEGR_BUFFERSIZE+*buff_occup]=qddoutput_1[i];
				qdoutput[i]=simp(accbuffer+i*INTEGR_BUFFERSIZE, stepsize)+qvinit[i];
				velbuffer[i*INTEGR_BUFFERSIZE+*buff_occup]=qdoutput[i];
			    qoutput[i]=simp(velbuffer+i*INTEGR_BUFFERSIZE, stepsize)+qinit[i];
		}
				
		(*buff_occup)++;
		break;

	case 3:
		for (i=0;i<njoints;i++){
				accbuffer[i*INTEGR_BUFFERSIZE+*buff_occup]=qddoutput_1[i];
				qdoutput[i]=simp3_8(accbuffer+i*INTEGR_BUFFERSIZE, stepsize)+qvinit[i];
				velbuffer[i*INTEGR_BUFFERSIZE+*buff_occup]=qdoutput[i];
			    qoutput[i]=simp3_8(velbuffer+i*INTEGR_BUFFERSIZE, stepsize)+qinit[i];
		}
				
		(*buff_occup)++;
		break;
	case 4:
		for (i=0;i<njoints;i++){
				accbuffer[i*INTEGR_BUFFERSIZE+*buff_occup]=qddoutput_1[i];
				qdoutput[i]=boolerule(accbuffer+i*INTEGR_BUFFERSIZE, stepsize)+qvinit[i];
				velbuffer[i*INTEGR_BUFFERSIZE+*buff_occup]=qdoutput[i];
			    qoutput[i]=boolerule(velbuffer+i*INTEGR_BUFFERSIZE, stepsize)+qinit[i];
		}
				
		(*buff_occup)++;
		break;
	case 5:
		for (i=0;i<njoints;i++){
				accbuffer[i*INTEGR_BUFFERSIZE+*buff_occup]=qddoutput_1[i];
				qdoutput[i]=fifth(accbuffer+i*INTEGR_BUFFERSIZE, stepsize)+qvinit[i];
				velbuffer[i*INTEGR_BUFFERSIZE+*buff_occup]=qdoutput[i];
			    qoutput[i]=fifth(velbuffer+i*INTEGR_BUFFERSIZE, stepsize)+qinit[i];
		}
				
		(*buff_occup)++;
		break;
	
	case 6:
		for (i=0;i<njoints;i++){
				accbuffer[i*INTEGR_BUFFERSIZE+*buff_occup]=qddoutput_1[i];
				qdoutput[i]=sixth(accbuffer+i*INTEGR_BUFFERSIZE, stepsize)+qvinit[i];
				sumacc[i]=qdoutput[i];
				velbuffer[i*INTEGR_BUFFERSIZE+*buff_occup]=qdoutput[i];
			    qoutput[i]=sixth(velbuffer+i*INTEGR_BUFFERSIZE, stepsize)+qinit[i];
				sumvel[i]=qoutput[i];
		}
				
		(*buff_occup)++;
		break;
	
	case 7:
		for (i=0;i<njoints;i++){
			/*Sliding window the elements are desplaced*/
				 memmove(accbuffer+i*INTEGR_BUFFERSIZE,accbuffer+i*INTEGR_BUFFERSIZE+1,(INTEGR_BUFFERSIZE-1)*sizeof(double));
			    accbuffer[i*INTEGR_BUFFERSIZE+6]=qddoutput_1[i];
				
			/*Computing the new vector*/	
				sumacc[i]+=sixth(accbuffer+i*INTEGR_BUFFERSIZE, stepsize)-fifth(accbuffer+i*INTEGR_BUFFERSIZE, stepsize);
				qdoutput[i]=sumacc[i];

		    /*Sliding window the elements are desplaced*/
				memmove(velbuffer+i*INTEGR_BUFFERSIZE,velbuffer+i*INTEGR_BUFFERSIZE+1,(INTEGR_BUFFERSIZE-1)*sizeof(double));
				velbuffer[i*INTEGR_BUFFERSIZE+6]=qdoutput[i];
				 
			/*Computing the new vector*/	 
				sumvel[i]+=sixth(velbuffer+i*INTEGR_BUFFERSIZE, stepsize)-fifth(velbuffer+i*INTEGR_BUFFERSIZE, stepsize);
				qoutput[i]=sumvel[i];
		}
				
		break;
	default:
		mexErrMsgTxt("Internal error of the integration buffer.");
}
return;
}

/**
 * Obtaining the size of the robot to create the needed variables
 * */

int load_robot(const char *robotfile, const char *robotname, int *size, mxArray **robot)
 /*PURPOSE:
         This is for obtaining robot object and size in order to dynamically create all the variables needed

     CALLING SEQUENCE:
         error=load_robot(robotfile, robotname, size, robot)


     INPUTS:
		  robotfile	   File name from where the robot object is loaded
		  robotname    Robot's variable name in the file
          robot        Robot variable in which the robot is returned
          size         Variable pointer where the number of joints are stored  

     OUTPUT:
       	 error         =0 if no error

*/
  {
   int nerror;
   MATFile *pmat;
   // Open MAT-file
   pmat=matOpen(robotfile, "r");
   if(pmat)
     {
      *robot = matGetVariable(pmat, robotname);
      if(*robot)
        {	  
         if(size)
            *size=mstruct_getint(*robot, 0, "n");	
         nerror=0;
        }
      else
         nerror=1;
      matClose(pmat);
     }
   else
      nerror=2;
   return(nerror);
  }

void free_robot(mxArray *robot)
 /*PURPOSE:
         This is for free the robot memory array

     CALLING SEQUENCE:
         free_robot(robot)


     INPUTS:
		  robot        Robot variable in which the robot is allocated


*/
{
      mxDestroyArray(robot); 
}

/**
 * Generating the trajectory
 * */
void trajectory(double *q, double *qd, double *qdd, double tsimul,int n_joints)
 /*PURPOSE:
         This is for generating an eight like trajectory in cartesian space by means of using sinusoidal curves
		 in joint space

     CALLING SEQUENCE:
         trajectory(q, qd, qdd,tsimul,n_joints)


     INPUTS:
		  tsimul	simulation time
          n_joint	number of links the robot has       

     OUTPUT:
       	 q          position per link at t= tsimul
		 qd			velocity per link at t=tsimul
		 qdd		acceleration per link at t=tsimul

*/
{
int i;
double Amplitude=0.1;
for (i=0;i<n_joints;i++){
    // trajectory initially proposed (qd[0] <> 0 and qd[1] <> 0)
	
	/*		q[i]=Amplitude*sin(2*M_PI*tsimul+(1.0/4.0)*i*M_PI);
	*		qd[i]=2*M_PI*Amplitude*cos(2*M_PI*tsimul+(1.0/4.0)*i*M_PI);
	*		qdd[i]=-4*pow(M_PI,2)*Amplitude*sin(2*M_PI*tsimul+(1.0/4.0)*i*M_PI);
	*/
	
    // modified trajectory by means of a cubic spline: -4sum(pisum(t^3)) + 6sum(pisum(t^2)) (qd[0] = 0 and qd[1] = 0 and q[0] = 0 and q[1] = 2*pi)
	q[i]=Amplitude*sin((-4*M_PI*tsimul*tsimul*tsimul + 6*M_PI*tsimul*tsimul) + i*M_PI/4);
	qd[i]=Amplitude*12*M_PI*tsimul*(1 - tsimul)*cos(4*M_PI*tsimul*tsimul*tsimul - 6*M_PI*tsimul*tsimul - M_PI*i/4);
	qdd[i]=Amplitude*(12*M_PI*(1-2*tsimul)*cos(4*M_PI*tsimul*tsimul*tsimul - 6*M_PI*tsimul*tsimul - M_PI*i/4) + 144*M_PI*M_PI*tsimul*tsimul*(tsimul - 1)*(tsimul - 1)*sin(4*M_PI*tsimul*tsimul*tsimul - 6*M_PI*tsimul*tsimul - M_PI*i/4));
	}
}

