
#include <iostream>
#include <list>
#include <vector>
#include <utility>
#include <iterator>
#include <functional>
#include <boost/iterator/zip_iterator.hpp>
#include <cstdlib>

#include "zip.h"
#include "algorithm.h" // std::my_copy_if

#include <cuda.h>
#include "cuda_utils/cuda_util.h"

// ------ Schelling -------

/** Definitions concerning the place type */
typedef std::pair<int, int> Position;
typedef int Place;
typedef std::vector<Position>  	  PositionList;
typedef std::vector<PositionList> PositionMatrix;
typedef std::vector<std::vector<Place> >  PlaceMatrix;

/** Constants from the simulation model */
namespace Schelling {
    const Place Free  = 0;
    const Place White = 1;
    const Place Black = 2;

    const int side = 500;

    const double freeP = 0.02;
    const double whiteP = 0.5;
}


class State {
public:
	int side_;
	PlaceMatrix matrix_;

private:
	inline int pmod(int i, int j) const {
		int m = i %j;
		if (m < 0) 	return m + j;
		return m;
	}

	static void randomCell (Place& inPlace) {
	    if(drand48() < Schelling::freeP) inPlace = Schelling::Free;
	    else if(drand48() < Schelling::whiteP) inPlace = Schelling::White;
	    else inPlace = Schelling::Black;
	}

public:
	explicit State(int inSize) :side_(inSize), matrix_(side_) {
        // reserve space for matrix columns
        // must use std::bind2nd -> http://www.cplusplus.com/reference/functional/mem_fun_ref/?kw=mem_fun_ref [Return Value]
        std::for_each (matrix_.begin(), matrix_.end(), std::bind2nd(std::mem_fun_ref(&std::vector<Place>::reserve), side_));
	}

    void init() {
        srand48(0);

        PlaceMatrix::iterator rows = matrix_.begin();
        PlaceMatrix::iterator rowsEnd = matrix_.end();

        // randomly initialize each cell according to model parameters
        for (; rows != rowsEnd; ++rows) {
            std::for_each(rows->begin(), rows->end(), randomCell);
        }
    }

	Place& operator() (int i, int j) {
		return matrix_[pmod(i, side_)][pmod(j, side_)];
	}
	const Place& operator() (int i, int j) const {
		return matrix_[pmod(i, side_)][pmod(j, side_)];
	}

};


PositionList moving(const State& inState, float inSimilarWanted) {
	// TODO
}

bool isFree(const Place& inPlace) {
    return Schelling::Free == inPlace;
}

PositionList freeCells(const State& inState) {
    PositionList freeCells;

    PlaceMatrix::const_iterator first = inState.matrix_.begin();
    PlaceMatrix::const_iterator end = inState.matrix_.end();
    PositionList::iterator current = freeCells.begin();

    for (; first != end; ++first) {
        current = std::my_copy_if(first->begin(), first->end(), first, end, current, isFree);
    }

    return freeCells;
}

/** Functor moving agents in the next state, and freeing their previous Place */
struct CopyMoves {

	const State& currentState;
	State& 		 nextState;

	explicit CopyMoves (const State& inCurrentState, State& inNextState)
		:currentState(inCurrentState), nextState(inNextState)
	{}

	void operator() (const boost::tuple<Position, Position>& inTuple) {
		nextState(inTuple.get<0>().first, inTuple.get<0>().second) = currentState(inTuple.get<1>().first, inTuple.get<1>().second);
        nextState(inTuple.get<1>().first, inTuple.get<1>().second) = Schelling::Free;
     }
};

State step(const State& inState) {
//	PositionList wantToMove  = moving(inState, 0.65);
    PositionList wantToMove;
	PositionList free 		 = freeCells(inState);

	std::random_shuffle(wantToMove.begin(), wantToMove.end());
	std::random_shuffle(free.begin(), free.end());

	zip_container<PositionList, PositionList> moves = zip(wantToMove, free);
	State nextState(inState);

	CopyMoves functor(inState, nextState);

	for_each(moves.begin(), moves.end(), functor);

	return nextState;
}

void simulation(State& inoutState, int nbSteps) {

	for (int i = nbSteps; 0 != i; --i) {
		std::cout << i << " steps left" << std::endl;
		step(inoutState);
	}

	std::cout << "done" << std::endl;
}


/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {

	// TODO
	State initialState(Schelling::side);
	initialState.init();
	simulation( initialState, 500);

	return 0;
}
