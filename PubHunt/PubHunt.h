/*
 * This file is part of the PubHunt distribution (https://github.com/kanhavishva/PubHunt).
 * Copyright (c) 2021 KV.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef KEYHUNTH
#define KEYHUNTH

#include <string>
#include <vector>
#ifdef WITHGPU
#include "GPU/GPUEngine.h"
#endif
#ifdef WIN64
#include <Windows.h>
#endif

#define CPU_GRP_SIZE (1024*2)

class PubHunt;

typedef struct {
	PubHunt* obj;
	int  threadId;
	bool isRunning;
	bool hasStarted;

	int  gridSizeX;
	int  gridSizeY;
	int  gpuId;

} TH_PARAM;


class PubHunt
{

public:

	PubHunt(const std::vector<std::vector<uint8_t>>& inputHashes, const std::string& outputFile);

	~PubHunt();

	void Search(std::vector<int> gpuId, std::vector<int> gridSize, bool& should_exit);
	void FindKeyGPU(TH_PARAM* p);

private:

	void output(const ITEM& result);
	bool isAlive(TH_PARAM* p);

	bool hasStarted(TH_PARAM* p);
	uint64_t getGPUCount();

	std::string formatThousands(uint64_t x);
	char* toTimeStr(int sec, char* timeStr);

	uint64_t counters[256];
	double startTime;

	bool endOfSearch;
	int nbGPUThread;
	int nbFoundKey;

	std::string outputFile;
	uint32_t *hash160;
	int numHash160;

	uint32_t maxFound;

#ifdef WIN64
	HANDLE ghMutex;
#else
	pthread_mutex_t  ghMutex;
#endif

};

#endif // KEYHUNTH
