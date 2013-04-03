package fr.isima.random


/**
 *  @author Jonathan Passerat-Palmbach
 *
 */
class ThreadLocalRandom(override val self: java.util.concurrent.ThreadLocalRandom ) extends scala.util.Random {

  /** Creates a new random number generator. */
  def this() = this(java.util.concurrent.ThreadLocalRandom.current())

  /** disabled for random stream distribution safety */
  override def setSeed(seed: Long) = { }

}

// removed companion object as well
