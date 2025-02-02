PFC = python3 -m numpy.f2py
LDFLAGS = -L/opt/local/lib
LIBS = -lblas

dot := .
empty :=
PYVER = $(word 2, $(subst $(dot),$(empty),$(basename $(shell python3 --version))))
UNAME = $(shell uname | tr '[:upper:]' '[:lower:]')

all: lbfgs cgf file_utility

lbfgs: lbfgs.f
	$(PFC) -c $< -m $@ --backend meson
ifeq ($(UNAME),darwin)
	install_name_tool -add_rpath /opt/local/lib/libgcc lbfgs.cpython-$(PYVER)-darwin.so
endif

cgf: cgsearch.f cgfam.f
	$(PFC) -c $^ -m $@ --backend meson $(LDFLAGS) $(LIBS)
ifeq ($(UNAME),darwin)
	install_name_tool -add_rpath /opt/local/lib/libgcc cgf.cpython-$(PYVER)-darwin.so
endif

file_utility: file_utility.f90
	$(PFC) -c $< -m $@ --backend meson
ifeq ($(UNAME),darwin)
	install_name_tool -add_rpath /opt/local/lib/libgcc file_utility.cpython-$(PYVER)-darwin.so
endif

clean: 
	rm -f *.cpython-$(PYVER)-$(UNAME).so
