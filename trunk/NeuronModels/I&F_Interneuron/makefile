
compiler := gcc
CXX := ${compiler}

source_model_file := _borisInterneuronECv3.c.boc_6i

tables_generator := TableGenerator.c
tables_executable := TableGenerator.exe

.PHONY : all

all : $(source_model_file) $(tables_generator)
	@echo
	@echo ------------------ creating neuron model file from $(source_model_file)
	@echo
	@cp $(source_model_file) tab2cfg.c
	$(compiler) -o $(tables_executable) $(tables_generator)
	$(tables_executable)
	@echo
	@echo ------------------ cleaning auxiliar files
	@echo
	@rm -f tab2cfg.c $(tables_executable)
