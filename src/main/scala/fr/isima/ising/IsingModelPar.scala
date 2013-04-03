package fr.isima.ising

import java.util.Random


object IsingModelPar {

  trait Lattice {
    def lattice : IndexedSeq[((Int, Int), Boolean)]
    def size: Int
  }

  def lattice(latticeSize : Int, threshold : Double)(implicit rng: Random) =
    new Lattice {
      val size = latticeSize
      val lattice = (0 to latticeSize * latticeSize - 1).par.map{i => (i % latticeSize -> i / latticeSize) -> (rng.nextDouble < threshold)}.toIndexedSeq
    }

  implicit def bool2double (b : Boolean) = if (b) 1.0 else -1.0
  implicit def lattice2IndexedSeq (l : Lattice) = l.lattice
}

import IsingModelPar._

class IsingModelPar {

  val kT = 2.0

  /**
   * Implicitly converts Boolean to their corresponding
   * integer value IN OUR CASE (false -> -1.0, true -> 1.0).
   */
  implicit def bool2double (b : Boolean) = if (b) 1.0 else -1.0
  
  
  def magnetization(lattice: Lattice) =
    lattice.map{case (_,s)=> s: Double}.reduce{(s1, s2) => s1 + s2}

  def processSpin(lattice: Lattice, inSpin : ((Int, Int), Boolean))(implicit rng: Random) = {
    val ((x, y), spin) = inSpin

    // compute energy of Von Neumann neighbourhood

    import lattice.size

    val energy: Double =
      lattice ( (x * size) + ( ((y - 1) * size) % size) )._2 +
    lattice ( (x * size) + ( ((y + 1) * size) % size) )._2 +
    lattice ( ( ((x - 1) * size) % size) + y )._2 +
    lattice ( ( ((x + 1) * size) % size) + y )._2 *
    lattice (y + x * size)._2 * -1

    // test metropolis criterion
    val deltaE = -2 * energy
    if ((deltaE <= 0) || (rng.nextDouble <= math.exp(-deltaE / kT) ) ) ((x, y), !spin)
    else inSpin
  }

  def isEven (x: Int, y: Int) = x % 2 == y % 2
  def isOdd (x: Int, y: Int) = !isEven(x, y)

  def processLattice(_lattice: Lattice)(implicit rng: Random) =
  
  new Lattice {
    val size = _lattice.size
    val lattice = 
      IndexedSeq.concat (
        _lattice.filter{case((x, y), _) => isEven(x, y)}.par.map(spin => processSpin(_lattice, spin)).toIndexedSeq,
        _lattice.filter{case((x, y), _) => isOdd(x, y)}.par.map(spin =>  processSpin(_lattice, spin)).toIndexedSeq
      )
  }
  
}
