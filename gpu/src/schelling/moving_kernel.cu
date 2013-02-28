namespace Schelling {
	const int neighbourhoodSize = 2;
}

__device__ inline bool isSimilarNeighbour(int* inFlatPosTab, const int inElementId, const int i, const int j, const int inTabSide) {
	return inFlatPosTab[inElementId] != 0 && inFlatPosTab[ i * inTabSide + j] == inFlatPosTab[inElementId];
}


__global__ void movingKernel (int* inFlatPosTab, int* outMovingTab, const int inTabSide) {
	int globalId = threadIdx.x + blockDim.x * blockIdx.x;

	if (globalId < inTabSide * inTabSide) {

		int neighboursCount;

		for (int i = -Schelling::neighbourhoodSize; i <= Schelling::neighbourhoodSize; ++i) {
			for (int j = -Schelling::neighbourhoodSize; j <= Schelling::neighbourhoodSize; ++j) {
				if ( isSimilarNeighbour(inFlatPosTab, globalId, i, j, inTabSide)  )	++neighboursCount;
			}
		}

		outMovingTab[globalId] = neighboursCount;
	}
}
