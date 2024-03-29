# EDLUT Spiking Neural Network Simulator

## Lastest news:

### EDLUT v3.0 major changes:
* EDLUT can now be easyly configured using cmake. 
* A new python interface has been implemented.
* New neuron models and learning rules have been implemented.
* Performance has been notably increased. 
* Some bugs fixed

### EDLUT v2.0 major changes:
* Time-driven cell models can now run on CUDA GPUs.
* Some pieces of code of the simulation kernel have been parallelized using OpenMP. 
* A simulated robot arm and several functions for neurorobotics integration are now included in the arm_robot_simulator folders. 
* Autoconfigure for Mac OS X and Linux has been added in order to simplify EDLUT compilation. 
* Performance has been notably increased. 
* Some bugs fixed

### EDLUT v1.0 major changes: 
* Simulation of Time-driven cell models and hybrid (time & event-driven) networks. 
* Implementation of Spike Time-Dependent Plasticity (STDP). 
* Compilable as a Matlab MEX function or Simulink S-function module. 
* Leaky-Integrate and Fire (LIF) and Spike-Response Models (SRM) implemented as time-driven cell models. 
* Lots of minor bugs fixed.

Get the last stable version of EDLUT source-code from this repository now.

We encourage all researchers interested in using EDLUT to have a look to the list of related papers, where you can find details about the evolution of EDLUT and some of the last robotic systems where EDLUT has been involved.

## EDLUT (Event-Driven simulator based on Look-Up-Tables)

The EDLUT (Event-Driven simulator based on Look-Up-Tables) is an advanced tool that allows the simulation of biologically plausible spiking cell models by using two different strategies: time-driven and event-driven based on look-up tables. In this way, EDLUT can highly speed up the simulation process by avoiding the resolution of the differential equations which usually regulate the evolution of the biological system state.

More detailed information about EDLUT simulation process can be found in the EDLUT Brief Description or the EDLUT presentation. If you are interested in scientific results using EDLUT, they can be found in Related Publications.

## Research Goals

EDLUT is a tool for studying the computational principles of neural systems and eventually contributing to reveal how different functionalities of the Brain and Central Nervous System are based on cell and topology properties.

We adopt the attitude of engineers: “I understand how it works when I build it”.

Investigating and creating models of nervous subsystems requires more than the simulation engine itself (EDLUT). It also needs cell models, network models, functional working hypothesis, etc. For this purpose, it is necessary an interdisciplinary research effort with contributions of neurophysiology groups, biological computing, cognitive systems, biology modelers, efficient computing, etc. In SpikeFORCE, SENSOPAC and REALNET several interdisciplinary cross-enriching collaborations are taking place for building biologically plausible models of neural subsystems such as the cerebellum, Inferior Olive, Cuneate Nucleuous, etc.

## Development Team and Collaborators

The original EDLUT has been developed at the University of Granada (Dept. of Computer Architecture and Technology). This project has been developed by J. Garrido, R. R. Carrillo, F. Naveros and N. R. Luque at the E. Ros' lab.

Now EDLUT has been released as Open Source facilitated by the OSL “Oficina de Software Libre” through the advice of J.J. Merelo of the University of Granada. This means that any other development effort can be done by any other member or the research community.

we have collaborated and are collaborating with different research groups: 
* University of Pavia (Egidio D’Angelo and Sergio Solinas). 
* University of Pierre and Marie Curie at Paris (Angelo Arleo, Luca Leonardo Bologna and Jean Baptiste Passot). 
* University of Erasmus (Chris de Zeeuw and Jornt de Gruijl). 
* University of Lund (Henrik Jörntell and Carl Fredrik Ekerot). 
* SICS (Martin Nilson). 
* Other researchers such as Boris Barbour (CNRS), Olivier Coenen and Mike Arnold.

Nevertheless, the final goal of understanting the computational principles of the Central Nervous System and how they are related with cell and topological properties is a medium and long term target which requires a continuous and international effort. Therefore any further collaboration is welcome.

Currently, we are improving performances and usability of EDLUT, so if you think that you could help in this issue, or you only have doubts about how to use EDLUT in your own simulations, don't hesitate to send an email to Jesús Garrido.

## Current Status

Currently we have abstracted different cell models such a Granule cell, Golgi cell, Purkinje cell, Hodgkin and Huxley model, etc. Some of them are represented as simple integrate and fire models and some include inherent dynamics (such as active ion conductances) that allow studying how specific cell properties impact system functionality.

We have done system models and networks of several hundred thousand cells (a simplified cerebellum model) on a conventional computer and we have done several hundred million simulations of 5 Kneurons to characterize sub-network dynamics.

Scientific results can be found in the related scientific papers and others currently under revision.

As future work we plan to interface the EDLUT simulation engine with other simulation tools widely used, such as NEURON.

Therefore the important message is that EDLUT allows efficient simulations of medium and large scale networks on conventional computers thanks to the event-driven simulation strategy based on Look-up-tables (of pre-compiled neuron models) which avoid intensive calculations during network simulations.

Further research on this issue and related topics will require intensive collaborations at international and interdisciplinary levels. EDLUT is open software; therefore it can be adapted or further developed by different research institutes. The EDLUT research core group at University of Granada (Eduardo Ros, Richard Carrillo and Jesus Garrido) are open to collaborations along this line.

## Acknowledgment

The development of the EDLUT platform has been supported by three EU grants, SpikeFORCE (IST-2001-35271), SENSOPAC (Sensorymotor structuring of Perception and Action for emerging cognition) (IST-028056) and REALNET (Realistic Real-time Networks: computation dynamics in the cerebellum) (IST-270434)
