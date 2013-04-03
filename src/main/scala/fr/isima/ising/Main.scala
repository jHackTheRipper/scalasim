package fr.isima.ising

object Main {

  def time(f: => Unit) = {
    val start = System.currentTimeMillis
    f
    System.currentTimeMillis - start
  }
  
  /**
   * @param args the command line arguments
   */
  def main(argv: Array[String]): Unit = {

    val N  = java.lang.Integer.parseInt(argv(0))
    val threshold = java.lang.Double.parseDouble(argv(1))
    val iters = java.lang.Integer.parseInt(argv(2))


    implicit val rng = new scala.util.Random
    //implicit val rng = new fr.isima.random.ThreadLocalRandom
    (1 to N).par.map {
      i =>
      rng.nextDouble * rng.nextDouble * rng.nextDouble * rng.nextDouble
    }
    
    
//    // change according to actual parallelization
//      import IsingModelSeq._
//    import IsingModelPar._
//    //  import IsingModelCL._
//
//    var initState = lattice(N, threshold)   
  // change according to actual parallelization
//    val model = new IsingModelSeq
//    val model = new IsingModelPar
  //    val model = new IsingModelCL
  
 //   println( "Execution time: " + time ( () => {
//          var lat1 = model.processLattice(initState)
//          for (i <- 0 to iters) {
//           lat1 = model.processLattice(lat1)
//          }
//        }
//     ) + " ms"
//    )
    
//    //   println("Magnetization: " + model.magnetization(lat1))
//    
//    // can't make it work!
//    // (0 until iters).foldLeft(initState)( (acc : Lattice, cur : Int) => model.processLattice(acc) )
//    
  }

}
