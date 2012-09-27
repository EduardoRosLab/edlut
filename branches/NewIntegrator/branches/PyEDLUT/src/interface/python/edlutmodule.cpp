/***************************************************************************
 *                           edlutmodule.cpp                               *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
 *                                                                         *
 * email                : jgarrido@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "Python.h"

#include "../../../include/simulation/Simulation.h"

#include "../../../include/communication/ArrayInputSpikeDriver.h"
#include "../../../include/communication/ArrayOutputSpikeDriver.h"

#include "../../../include/spike/EDLUTFileException.h"
#include "../../../include/spike/EDLUTException.h"

#include <limits>

static Simulation * CurrentSimulation = 0;
static ArrayInputSpikeDriver * InputDriver = 0;
static ArrayOutputSpikeDriver * OutputDriver = 0;

static PyObject *EDLUTError;

PyDoc_STRVAR(edlut_loadnet_doc,
"loadnet(network,weights[,timedrivenstep])\n\
\n\
Load the network and initialize the simulation.");

static PyObject *
edlut_loadnet(PyObject *self, PyObject *args, PyObject *keywds)
{
    char * NetworkFile, * WeightsFile;
    double TimeDrivenStep = -1;

    static char *kwlist[] = {"network", "weights", "timedrivenstep", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss|d:loadnet", kwlist, &NetworkFile, &WeightsFile, &TimeDrivenStep))
        return NULL;

    try{
    	// Create and initialize simulation
    	if (CurrentSimulation!=0){
    		delete CurrentSimulation;
    	}
    	CurrentSimulation = new Simulation((const char *) NetworkFile, (const char *) WeightsFile, numeric_limits<double>::max(), -1);
		if (TimeDrivenStep!=-1){
			CurrentSimulation->SetTimeDrivenStep(TimeDrivenStep);
		}

		// Create and initialize input array
		if (InputDriver!=0){
			delete InputDriver;
		}
		InputDriver = new ArrayInputSpikeDriver();
		CurrentSimulation->AddInputSpikeDriver(InputDriver);

		// Create and initialize output array
		if (OutputDriver!=0){
			delete OutputDriver;
		}
		OutputDriver = new ArrayOutputSpikeDriver();
		CurrentSimulation->AddOutputSpikeDriver(OutputDriver);

		// Initialize simulation
		CurrentSimulation->InitSimulation();
    } catch (EDLUTFileException Exc){
    	PyErr_SetString(EDLUTError, Exc.GetErrorMsg());
    	return NULL;
	} catch (EDLUTException Exc){
		PyErr_SetString(EDLUTError, Exc.GetErrorMsg());
	    return NULL;
	}

	Py_INCREF(Py_None);
	return Py_None;
}


PyDoc_STRVAR(edlut_injectspikes_doc,
"injectspikes(timefiring,cellfiring)\n\
\n\
Insert input activity into the network.");

static PyObject *
edlut_injectspikes(PyObject *self, PyObject *args, PyObject *keywds)
{

	PyObject *time_list, *cell_list;
	static char *kwlist[] = {"spiketimes", "firingcells", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO:injectspikes", kwlist, &time_list, &cell_list))
		return NULL;

	if (!PySequence_Check(time_list)) {
		PyErr_SetString(EDLUTError, "Expected sequence in spiketimes");
	    return NULL;
	}

	if (!PySequence_Check(cell_list)) {
		PyErr_SetString(EDLUTError, "Expected sequence in firingcells");
		return NULL;
	}

	PyObject *item_time, *item_cell;
	int time_size, cell_size;

	double * spike_time_array;
	long int * spike_cell_array;

	time_size = PyObject_Length(time_list);
	cell_size = PyObject_Length(cell_list);

	if (time_size!=cell_size){
		PyErr_SetString(EDLUTError, "Spiketimes and firingcells sequences must have the same length");
		return NULL;
	}

	/* create a dynamic C array of integers */
	spike_time_array = (double *) new double [time_size];
	spike_cell_array = (long int *) new long int [cell_size];


	for (int index = 0; index < time_size; index++) {
		/* get the element from the list/tuple */
		item_time = PySequence_GetItem(time_list, index);
	    /* we should check that item != NULL here */
	    /* make sure that it is a Python double */
	    if (!PyFloat_Check(item_time)) {
	      delete [] spike_time_array;  /* free up the memory before leaving */
	      delete [] spike_cell_array;
	      PyErr_SetString(EDLUTError, "Expected sequence of float in spiketimes parameter");
	      return NULL;
	    }

	    /* get the element from the list/tuple */
		item_cell = PySequence_GetItem(cell_list, index);
		/* we should check that item != NULL here */
		/* make sure that it is a Python double */
		if (!PyInt_Check(item_cell)) {
		  delete [] spike_time_array;  /* free up the memory before leaving */
		  delete [] spike_cell_array;
		  PyErr_SetString(EDLUTError, "Expected sequence of int in firingcells parameter");
		  return NULL;
		}

	    /* assign to the C array */
	    spike_time_array[index] = PyFloat_AsDouble(item_time);
	    spike_cell_array[index] = PyInt_AsLong(item_cell);
	}


	// Load inputs
	InputDriver->LoadInputs(CurrentSimulation->GetQueue(),CurrentSimulation->GetNetwork(),time_size,spike_time_array,spike_cell_array);

	delete [] spike_time_array;
	delete [] spike_cell_array;

	Py_INCREF(Py_None);
    return Py_None;

}

PyDoc_STRVAR(edlut_getoutput_doc,
"outputactivity = getoutput()\n\
\n\
Get the output activity of the network.");

static PyObject *
edlut_getoutput(PyObject *self, PyObject *args, PyObject *keywds)
{
	double * OutputSpikeTimes;
	long int * OutputSpikeCells;

	// Get outputs and print them
	int OutputNumber = OutputDriver->GetBufferedSpikes(OutputSpikeTimes,OutputSpikeCells);

	PyObject * pylist_time = PyList_New(OutputNumber);
	PyObject * pylist_cell = PyList_New(OutputNumber);

	if (OutputNumber>0){

		for (unsigned int i = 0; i < OutputNumber; i++) {
			/* convert resulting array to PyObject */
			PyObject * item_time = PyFloat_FromDouble(OutputSpikeTimes[i]);
			PyObject * item_cell = PyInt_FromLong(OutputSpikeCells[i]);

			PyList_SetItem(pylist_time, i, item_time);
			PyList_SetItem(pylist_cell, i, item_cell);
		}

		delete [] OutputSpikeTimes;
		delete [] OutputSpikeCells;
	}

	PyObject * pylist_total = PyList_New(2);
	PyList_SetItem(pylist_total, 0, pylist_time);
	PyList_SetItem(pylist_total, 1, pylist_cell);

	return pylist_total;

}

PyDoc_STRVAR(edlut_simulateslot_doc,
"simulateslot(timestep)\n\
\n\
Simulate up to the specified time.");

static PyObject *
edlut_simulateslot(PyObject *self, PyObject *args, PyObject *keywds)
{
	static char *kwlist[] = {"time", NULL};

	double simulation_time;

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "d:simulateslot", kwlist, &simulation_time))
        return NULL;

	// Simulate until CurrentTime+StepTime
	CurrentSimulation->RunSimulationSlot(simulation_time);

    Py_INCREF(Py_None);
    return Py_None;
}

PyDoc_STRVAR(edlut_finalize_doc,
"finalize()\n\
\n\
Finalize the simulation and release the memory");

static PyObject *
edlut_finalize(PyObject *self, PyObject *args, PyObject *keywds)
{
	if (CurrentSimulation!=0){
		delete CurrentSimulation;
		CurrentSimulation = 0;
	}

	if (InputDriver!=0){
		delete InputDriver;
		InputDriver = 0;
	}

	if (OutputDriver!=0){
		delete OutputDriver;
		OutputDriver = 0;
	}

    Py_INCREF(Py_None);
    return Py_None;
}


/* ---------- */


/* List of functions defined in the module */

static PyMethodDef edlut_methods[] = {
    {"loadnet",	(PyCFunction) edlut_loadnet,	METH_VARARGS | METH_KEYWORDS, edlut_loadnet_doc},
    {"injectspikes", (PyCFunction) edlut_injectspikes,	METH_VARARGS | METH_KEYWORDS,	edlut_injectspikes_doc},
    {"getoutput", (PyCFunction) edlut_getoutput,	METH_VARARGS | METH_KEYWORDS,	edlut_getoutput_doc},
    {"simulateslot", (PyCFunction) edlut_simulateslot,	METH_VARARGS | METH_KEYWORDS,	edlut_simulateslot_doc},
    {"finalize", (PyCFunction) edlut_finalize,	METH_VARARGS | METH_KEYWORDS,	edlut_finalize_doc},
    {NULL,	NULL, 0, NULL}           /* sentinel */
};

PyDoc_STRVAR(module_doc,
"This is a Python interface for EDLUT");

/* Initialization function for the module (*must* be called initxx) */

PyMODINIT_FUNC
initedlut(void)
{
    PyObject *m;

    /* Create the module and add the functions */
    m = Py_InitModule3("edlut", edlut_methods, module_doc);
    if (m == NULL)
        return;

    if (EDLUTError == NULL) {
    	EDLUTError = PyErr_NewException("edlut.error", NULL, NULL);
    	if (EDLUTError == NULL)
    		return;
    }
    Py_INCREF(EDLUTError);
    PyModule_AddObject(m, "error", EDLUTError);
}
