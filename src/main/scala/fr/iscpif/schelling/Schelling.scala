package fr.iscpif.schelling

import util.Random
import math._
import collection.mutable.ArrayBuffer
import Parameters.neighborhoodSize

object Schelling extends App { 

  implicit val rng = new Random
  
  trait Place
  case object Free extends Place
  case object White extends Place
  case object Black extends Place
 
  def pmod(i: Int, j: Int) = {
    val m = i % j
    if(m < 0) m + j else m
  }

  class State(val matrix: Seq[Seq[Place]]) extends Iterable[Place] {
    def side = matrix.size
    def iterator = matrix.iterator.flatten
    def apply(i: Int)(j: Int) = matrix(pmod(i, side))(pmod(j, side))
    def cells = matrix.zipWithIndex.flatMap{ case(l, i) => l.zipWithIndex.map{ case(c, j) => ((i, j), c) } }
    override def toString = matrix.toString
  }

  // Randomly draw a cell type given the proportions
  def randomCell(freeP: Double, whiteP: Double)(implicit rng: Random) =
    if(rng.nextDouble < freeP) Free
    else if(rng.nextDouble < whiteP) White else Black
  
  // Generate randomly an initial state
  def initial(side: Int, freeP: Double, whiteP: Double)(implicit rng: Random): State =
    new State(Seq.fill(side, side)(randomCell(freeP, whiteP)))

  // Compute the proportion of similar neighbors in a neighborhood of neighborhoodSize
  def similarNeighbors(state: State, i: Int, j: Int) = {
    val n = neighbors(state, i, j).filter(_ != Free).toList
    n.count{ _ == state(i)(j) } / n.size.toDouble
  }

  // Compute the list of coordinates of the agents that want to move
  def moving(state: State, similarWanted: Double) =
    state.cells.filter{case (_, c) => c != Free}.flatMap {
      case((i, j), _) =>
        if(similarNeighbors(state, i, j) < similarWanted) List((i, j)) else List.empty
    }

  def neighbors(state: State, i: Int, j: Int) = 
    for {
      oi <- -neighborhoodSize to neighborhoodSize
      oj <- -neighborhoodSize to neighborhoodSize
      if(oi != 0 || oj != 0)
    } yield state(i + oi)(j + oj)

  def freeCells(state: State) = state.cells.filter{ case(_, c) => c == Free }.unzip._1
  
  def step(state: State)(implicit rng: Random) = {
    val wantToMove = moving(state, 0.65)
    val free = freeCells(state)

    val moves = rng.shuffle(wantToMove) zip rng.shuffle(free)

    val newMatrix = ArrayBuffer.tabulate(state.side, state.side)((i, j) => state(i)(j))
    for(((fromI, fromJ), (toI, toJ)) <- moves) { 
      newMatrix(toI)(toJ) = state(fromI)(fromJ)
      newMatrix(fromI)(fromJ) = Free
    }

    new State(newMatrix)
  }
 
  def simulation(state: State = initial (Parameters.side, Parameters.freeP, Parameters.whiteP), nbStep: Int = Parameters.steps): State = {
    println(nbStep + " steps left")
    if( 0 == nbStep )   state
    else                simulation(step(state), nbStep - 1)
  }

  (0 until 10) foreach ( _ => simulation() )

}
