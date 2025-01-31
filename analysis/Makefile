PFC = python3 -m numpy.f2py
LDFLAGS = -L/opt/local/lib
LIBS = -lblas

all: lbfgs cgf file_utility

lbfgs: lbfgs.f
	$(PFC) -c $< -m $@ --backend meson
	install_name_tool -add_rpath /opt/local/lib/libgcc lbfgs.cpython-312-darwin.so

cgf: cgsearch.f cgfam.f
	$(PFC) -c $^ -m $@ --backend meson $(LDFLAGS) $(LIBS)
	install_name_tool -add_rpath /opt/local/lib/libgcc cgf.cpython-312-darwin.so

file_utility: file_utility.f90
	$(PFC) -c $< -m $@ --backend meson
	install_name_tool -add_rpath /opt/local/lib/libgcc file_utility.cpython-312-darwin.so

clean: 
	rm -f *.cpython*.so
