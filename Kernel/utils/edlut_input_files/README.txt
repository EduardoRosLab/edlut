#How to run EDLUT directly using configuration files. Once edlut have been installed, the edlutkernel code should be available in your path. You can run edlut using the terminal.

------->>>>>>> edlutkernel -time 1 -nf network_file.cfg -wf weight_file.cfg -if Input_spikes.cfg -ifc Input_currents.cfg -openmp 1

where:
	-time X is the execution time in seconds
	-nf X is the configuration network file
	-wf X is the configuration sinaptic weight file
	-if X is the input spike configuration file
	-ifc X is the input current configuration file
	-openmp X is the number of CPU threads used


