#---------------------------------------------------------------------
# Makefile for PubHunt
#

SRC = IntGroup.cpp Main.cpp Random.cpp Timer.cpp \
      Int.cpp IntMod.cpp Utils.cpp PubHunt.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
        IntGroup.o Main.o Random.o Timer.o Int.o \
        IntMod.o PubHunt.o Utils.o GPU/GPUEngine.o)

CXX        = g++
CUDA       = /usr/local/cuda-11.0
CXXCUDA    = /usr/bin/g++
NVCC       = $(CUDA)/bin/nvcc
# nvcc requires joint notation w/o dot, i.e. "5.2" -> "52"
ccap       = $(shell echo $(CCAP) | tr -d '.')


CXXFLAGS   =  -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I. -I$(CUDA)/include
LFLAGS     = -lpthread -L$(CUDA)/lib64 -lcudart -lcurand

#--------------------------------------------------------------------

$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu


$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: PubHunt

PubHunt: $(OBJET)
	@echo Making PubHunt...
	$(CXX) $(OBJET) $(LFLAGS) -o PubHunt

$(OBJET): | $(OBJDIR) $(OBJDIR)/GPU

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p GPU

clean:
	@echo Cleaning...
	@rm -f obj/*.o
	@rm -f obj/GPU/*.o

