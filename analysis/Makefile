PFC = python3 -m numpy.f2py
LDFLAGS = -L/opt/local/lib
LIBS = -lblas

all: lbfgs cgf file_utility

lbfgs: lbfgs.f
	$(PFC) $< -h lbfgs.pyf -m $@
	$(PFC) -c lbfgs.pyf $<

cgf: cgsearch.f cgfam.f
	$(PFC) $^ -h cgf.pyf -m $@
	$(PFC) -c cgf.pyf $(LDFLAGS) $(LIBS) $^

file_utility: file_utility.f90
	$(PFC) $< -h file_utility.pyf -m $@
	$(PFC) -c file_utility.pyf $<

clean: 
	rm -f *.pyf
