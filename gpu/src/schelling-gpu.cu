/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include <iostream>
#include <cuda.h>
#include <list>
#include <vector>
#include <utility>
#include <iterator>
#include <boost/iterator/zip_iterator.hpp>

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 * TODO replace with cuda_utils
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


/** Definitions concerning the place type */
typedef std::pair<int, int> Position;
typedef int Place;
typedef std::vector<Position>  	  PositionList;
typedef std::vector<PositionList> PositionMatrix;

/** Constants from the simulation model */
const Place Free  = 0;
const Place White = 1;
const Place Black = 2;

const int side = 500;

class State {
	std::vector<std::vector<Place> > matrix_;

	int side_;

private:
	inline int pmod(int i, int j) const {
		int m = i %j;
		if (m < 0) 	return m + j;
		return m;
	}

public:
	explicit State(int inSize) :side_(inSize) {}
	Place& operator() (int i, int j) {
		return matrix_[pmod(i, side_)][pmod(j, side_)];
	}
	const Place& operator() (int i, int j) const {
		return matrix_[pmod(i, side_)][pmod(j, side_)];
	}

};

/** Equivalent of zip function from Scala
 *  Adapted from http://stackoverflow.com/a/8511125/470341 */
template <typename C1, typename C2>
class zip_container {
    C1* c1;
    C2* c2;


public:

    // CUDA does not allow C++11 constructs yet
    // thus had to get rid of decltype
    typedef typename C1::iterator tuple_1;
    typedef typename C2::iterator tuple_2;

    typedef boost::tuple<
        tuple_1,
        tuple_2
    > tuple;

    zip_container(C1& c1, C2& c2) : c1(&c1), c2(&c2) {}

    typedef typename boost::zip_iterator<tuple> iterator;

    iterator begin() const { return boost::make_zip_iterator(boost::make_tuple(c1->begin(), c2->begin() )); }
    iterator end()   const { return boost::make_zip_iterator(boost::make_tuple(c1->end(),   c2->end()   ));   }

    inline std::size_t size() { return end() - begin(); }
};

template <typename C1, typename C2>
zip_container<C1, C2> zip(C1& c1, C2& c2) {
    return zip_container<C1, C2>(c1, c2);
}



PositionList moving(const State& inState, float inSimilarWanted) {
	// TODO

//	unsigned int nbBlock = ceil ();
//	movingKernel<<< nbBlocks, 256 >>> ();
}

PositionList freeCells(const State& inState) {
	// TODO
}

struct CopyMoves {

	const State& currentState;
	State& 		 nextState;

	explicit CopyMoves (const State& inCurrentState, State& inNextState)
		:currentState(inCurrentState), nextState(inNextState)
	{}

	void operator() (const zip_container<PositionList, PositionList>::tuple& inTuple) {
		nextState(inTuple.get<0>()->first, inTuple.get<0>()->second) = currentState(inTuple.get<1>()->first, inTuple.get<1>()->second);
		nextState(inTuple.get<1>()->first, inTuple.get<1>()->second) = 0;
	}
};

void step(State& inoutState) {
	PositionList wantToMove  = moving(inoutState, 0.65);
	PositionList free 		 = freeCells(inoutState);

	std::random_shuffle(wantToMove.begin(), wantToMove.end());
	std::random_shuffle(free.begin(), free.end());

	zip_container<PositionList, PositionList> moves = zip(wantToMove, free);
	State nextState(inoutState);

	CopyMoves functor(inoutState, nextState);

//	zip_container<PositionList, PositionList>::iterator first = moves.begin();
//	zip_container<PositionList, PositionList>::iterator end = moves.end();
//	for (; first != end; ++first) {
//		zip_container<std::vector<std::pair<int,int> >, std::vector<std::pair<int,int> > >::tuple = boost::make_tuple(first, end);
//	}

	for_each(moves.begin(), moves.end(), functor);
}

void simulation(State& inoutState, int nbSteps) {
	std::cout << nbSteps << " left" << std::endl;

	for (int i = nbSteps; 0 != i; --i) {
		step(inoutState);
	}

	std::cout << "done" << std::endl;
}

/**
 * Kernel processing the pre-filtered list of agents that want to move
 * @param inAgentsLists List of agents that have been stated as moving
 */
__global__ void movingKernel() {
	unsigned int globalId = threadIdx.x + blockDim.x * blockIdx.x;

	// TODO
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {

	// TODO
//	State initialState(, ,);
//	simulation( initialState, 500);

	return 0;
}
