/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
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

#include <device_atomic_functions.h>
#include <device_functions.h>

// ---------------------------------------------------------------------------------------

__device__ __noinline__ bool MatchHash160(uint32_t* _h, uint32_t* hash160)
{
	if (_h[0] == hash160[0] &&
		_h[1] == hash160[1] &&
		_h[2] == hash160[2] &&
		_h[3] == hash160[3] &&
		_h[4] == hash160[4]) {
		return true;
	}
	else {
		return false;
	}
}

__device__ void ComputeHash(uint64_t* keys, uint32_t* hash160, int numHash160, uint32_t maxFound, uint32_t* found)
{

	uint32_t hE[5];
	uint32_t hO[5];

	_GetHash160Comp(keys, 0, (uint8_t*)hE);
	_GetHash160Comp(keys, 1, (uint8_t*)hO);
	
	//for (int32_t i = 0; i < 32; i++) {
	//	printf("%02x", ((uint8_t*)keys)[i]);
	//}

	//printf(" ");
	//for (int32_t i = 0; i < 20; i++) {
	//	printf("%02x", ((uint8_t*)hE)[i]);
	//}
	//printf(" ");
	//for (int32_t i = 0; i < 20; i++) {
	//	printf("%02x", ((uint8_t*)hO)[i]);
	//}

	//printf("\n");

	uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t* pubKey = (uint32_t*)keys;

	for (int i = 0; i < numHash160; i++) {

		uint32_t* hash = hash160 + 5 * i;

		// match for even pubkey
		if (MatchHash160(hE, hash)) {
			uint32_t pos = atomicAdd(found, 1);

			if (pos < maxFound) {

				found[pos * ITEM_SIZE_A32 + 1] = tid;


				found[pos * ITEM_SIZE_A32 + 2] = 0x02;		// even
				found[pos * ITEM_SIZE_A32 + 3] = pubKey[0];
				found[pos * ITEM_SIZE_A32 + 4] = pubKey[1];
				found[pos * ITEM_SIZE_A32 + 5] = pubKey[2];
				found[pos * ITEM_SIZE_A32 + 6] = pubKey[3];
				found[pos * ITEM_SIZE_A32 + 7] = pubKey[4];
				found[pos * ITEM_SIZE_A32 + 8] = pubKey[5];
				found[pos * ITEM_SIZE_A32 + 9] = pubKey[6];
				found[pos * ITEM_SIZE_A32 + 10] = pubKey[7];

				found[pos * ITEM_SIZE_A32 + 11] = hE[0];
				found[pos * ITEM_SIZE_A32 + 12] = hE[1];
				found[pos * ITEM_SIZE_A32 + 13] = hE[2];
				found[pos * ITEM_SIZE_A32 + 14] = hE[3];
				found[pos * ITEM_SIZE_A32 + 15] = hE[4];
			}
		}

		// match for odd pubkey
		if (MatchHash160(hO, hash)) {
			uint32_t pos = atomicAdd(found, 1);

			if (pos < maxFound) {

				found[pos * ITEM_SIZE_A32 + 1] = tid;

				found[pos * ITEM_SIZE_A32 + 2] = 0x03;		// odd
				found[pos * ITEM_SIZE_A32 + 3] = pubKey[0];
				found[pos * ITEM_SIZE_A32 + 4] = pubKey[1];
				found[pos * ITEM_SIZE_A32 + 5] = pubKey[2];
				found[pos * ITEM_SIZE_A32 + 6] = pubKey[3];
				found[pos * ITEM_SIZE_A32 + 7] = pubKey[4];
				found[pos * ITEM_SIZE_A32 + 8] = pubKey[5];
				found[pos * ITEM_SIZE_A32 + 9] = pubKey[6];
				found[pos * ITEM_SIZE_A32 + 10] = pubKey[7];

				found[pos * ITEM_SIZE_A32 + 11] = hO[0];
				found[pos * ITEM_SIZE_A32 + 12] = hO[1];
				found[pos * ITEM_SIZE_A32 + 13] = hO[2];
				found[pos * ITEM_SIZE_A32 + 14] = hO[3];
				found[pos * ITEM_SIZE_A32 + 15] = hO[4];
			}
		}
	}
	__syncthreads();

}