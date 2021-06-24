#include "Timer.h"
#include "Int.h"
#include "Random.h"
#include "PubHunt.h"
#include "Utils.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <string.h>
#include <stdexcept>
#include <cassert>
#ifndef WIN64
#include <signal.h>
#include <unistd.h>
#endif

#define RELEASE "1.00"

using namespace std;
bool should_exit = false;

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------

void printUsage() {

	printf("PubHunt [-check] [-h] [-v] \n");
	printf("        [-gi GPU ids: 0,1...] [-gx gridsize: g0x,g0y,g1x,g1y, ...]\n");
	printf("        [-o outputfile] [inputFile]\n\n");
	printf(" -v                       : Print version\n");
	printf(" -gi gpuId1,gpuId2,...    : List of GPU(s) to use, default is 0\n");
	printf(" -gx g1x,g1y,g2x,g2y, ... : Specify GPU(s) kernel gridsize, default is 8*(MP number),128\n");
	printf(" -o outputfile            : Output results to the specified file\n");
	printf(" -l                       : List cuda enabled devices\n");
	printf(" -check                   : Check CPU and GPU kernel vs CPU\n");
	printf(" inputFile                : List of the hash160, one per line in hex format (text mode)\n\n");
	exit(0);

}
// ----------------------------------------------------------------------------


#ifdef WIN64
BOOL WINAPI CtrlHandler(DWORD fdwCtrlType)
{
	switch (fdwCtrlType) {
	case CTRL_C_EVENT:
		//printf("\n\nCtrl-C event\n\n");
		should_exit = true;
		return TRUE;

	default:
		return TRUE;
	}
}
#else
void CtrlHandler(int signum) {
	printf("\n\nBYE\n");
	exit(signum);
}
#endif

int main(int argc, const char* argv[])
{
	// Global Init
	Timer::Init();
	rseed(Timer::getSeed32());

	srand(time(NULL));

	// for Int check
	Int P, order;
	P.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
	order.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
	Int::InitK1(&order);
	Int::SetupField(&P);
	//

	vector<int> gpuId = { 0 };
	vector<int> gridSize;

	string outputFile = "Found.txt";

	std::vector<std::vector<uint8_t>> inputHashes;


	int a = 1;

	while (a < argc) {
		if (strcmp(argv[a], "-gi") == 0) {
			a++;
			getInts("gi", gpuId, string(argv[a]), ',');
			a++;
		}
		else if (strcmp(argv[a], "-v") == 0) {
			printf("%s\n", RELEASE);
			exit(0);
		}
		else if (strcmp(argv[a], "-check") == 0) {

			Int::Check();
#ifdef WITHGPU
			if (gridSize.size() == 0) {
				gridSize.push_back(-1);
				gridSize.push_back(128);
			}
#else
			printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif
			exit(0);
		}
		else if (strcmp(argv[a], "-l") == 0) {

#ifdef WITHGPU
			GPUEngine::PrintCudaInfo();
#else
			printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif
			exit(0);

		}
		else if (strcmp(argv[a], "-gx") == 0) {
			a++;
			getInts("gridSize", gridSize, string(argv[a]), ',');
			a++;
	}
		else if (strcmp(argv[a], "-o") == 0) {
			a++;
			outputFile = string(argv[a]);
			a++;
		}
		else if (strcmp(argv[a], "-h") == 0) {
			printUsage();
		}
		else if (a == argc - 1) {
			parseFile(string(argv[a]), inputHashes);
			a++;
		}
		else {
			printf("Unexpected %s argument\n", argv[a]);
			exit(-1);
		}

}

	if (gridSize.size() == 0) {
		for (int i = 0; i < gpuId.size(); i++) {
			gridSize.push_back(-1);
			gridSize.push_back(128);
		}
	}
	else if (gridSize.size() != gpuId.size() * 2) {
		printf("Invalid gridSize or gpuId argument, must have coherent size\n");
		exit(-1);
	}

	printf("\n");
	printf("PubHunt v" RELEASE "\n");
	printf("\n");
	printf("DEVICE       : %s\n", "GPU");
	if (1) {
		printf("GPU IDS      : ");
		for (int i = 0; i < gpuId.size(); i++) {
			printf("%d", gpuId.at(i));
			if (i + 1 < gpuId.size())
				printf(", ");
		}
		printf("\n");
		printf("GPU GRIDSIZE : ");
		for (int i = 0; i < gridSize.size(); i++) {
			printf("%d", gridSize.at(i));
			if (i + 1 < gridSize.size()) {
				if ((i + 1) % 2 != 0) {
					printf("x");
				}
				else {
					printf(", ");
				}
			}
		}
		printf("\n");
	}
	printf("NUM HASH160  : %llu\n", inputHashes.size());
	printf("OUTPUT FILE  : %s\n", outputFile.c_str());

#ifdef WIN64
	if (SetConsoleCtrlHandler(CtrlHandler, TRUE)) {

		PubHunt* v = new PubHunt(inputHashes, outputFile);

		v->Search(gpuId, gridSize, should_exit);
		delete v;
		printf("\n\nBYE\n");
		return 0;
	}
	else {
		printf("Error: could not set control-c handler\n");
		return 1;
	}
#else
	signal(SIGINT, CtrlHandler);

	PubHunt* v = new PubHunt(inputHashes, outputFile);

	v->Search(gpuId, gridSize, should_exit);
	delete v;
	return 0;
#endif
}