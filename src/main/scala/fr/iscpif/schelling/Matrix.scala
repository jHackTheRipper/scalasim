package fr.iscpif.schelling

import scalacl._

class Matrix(data: CLArray[Float], rows: Int, columns: Int) {
implicit val context = Context.best
   
			   def this(rows: Int, columns: Int) =
			      this(new CLArray[Float](rows * columns), rows, columns)
			      def this(n: Int) =
			      this(n, n)

			      def putProduct(a: Matrix, b: Matrix): Unit = {
				 assert(a.columns == b.rows)
				    assert(a.rows == rows)
				    assert(b.columns == columns)

				    kernel {
				       // This block will either be converted to an OpenCL kernel or cause compilation error
				       for (i <- 0 until rows; j <- 0 until columns) {
					  data(i * columns + j) = (0 until a.columns).map(k => {
						a.data(i * a.columns + k) * b.data(k * b.columns + j)
						}).sum
				       }
				    }
			      }

			   def putSum(a: Matrix, b: Matrix): Unit = {
			      assert(a.columns == b.columns && a.columns == columns)
				 assert(a.rows == b.rows && a.rows == rows)

				 kernel {
				    for (i <- 0 until rows; j <- 0 until columns) {
				       val offset = i * columns + j
					  data(offset) = a.data(offset) + b.data(offset)
				    }
				 }
			   }
			}
/*

   val n = 10
   val a = new Matrix(n)
   val b = new Matrix(n)
val out = new Matrix(n)

out.putProduct(a, b)

println(out.data)
*/
