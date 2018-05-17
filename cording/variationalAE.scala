package VAE
import breeze.linalg._
import math._
import scala.sys.process.Process
import java.io.{FileOutputStream=>FileStream,OutputStreamWriter=>StreamWriter}

//                          myu   -- myu^2 +  ...\_ KL_D
//     -- A -- S --myu      sigma -- sigma^2 +.../
// x <
//     -- A -- S --sigma
//                          z=myu+sig -- A -- T -- y

object VAE{

  val rand=new util.Random(0)
  def main(args:Array[String]){

    val mode = args(0)
    val ln   = args(1).toInt
    val dn   = args(2).toInt
    var myu_E =mode match{
      case "0" =>{
        val a = new Affine(1,1)
        val b = new Sigmoid()
        List(a,b)
      }
    }

    var sigma_E = mode match{
      case "0" => {
        val a1 = new Affine(1,1)
        val b1 = new Sigmoid()
        List(a1,b1)
      }
    }

    var decoder = mode match{
      case "0" => {
        val a2 = new Affine(1,1)
        val b2 = new Tanh()
        List(a2,b2)
      }
    }
    var err1_b = 0d;var err2_b = 0d;var err3_b = 0d;var err4_b = 0d

    var trainList = List[Double]()
    var testList = List[Double]()
    var train_inList = List[Double]()
    var test_inList = List[Double]()
  

    //start
    for (i <- 0 until ln){
      var start_a = System.currentTimeMillis 
      val trainin = createInput(dn)
      val testin  = createInput(dn)
      //training
      var err1    = 0d; var err2    = 0d
      var eError  = 0d; var dError  = 0d      
      var E_train = List[Double]()
      var D_train = List[Double]()
      
      for(x <- trainin){
        val ave = forwards(myu_E,x)
        val dis = forwards(sigma_E,x)

        val z = dis.map(_*rand.nextGaussian).zip(ave).map{case (a,b) => a + b}
        val y = forwards(decoder,z)

        err1 = mse(y,x)
        err2 = divergence(ave,dis)
        E_train ::= err1
        D_train ::= err2

        //backward
        val d = backwards(decoder,y.zip(x).map{
          case (a,b) => a - b
        })

        val aveD = d.zip(ave).map{case (a,b) => a+b}
        val tmp = dis.map(a => -1d/(a*a*2)).zip(dis).map{case (a,b) => a+b}
        val disD = d.zip(tmp).map{case (a,b) => a+b}
        backwards(myu_E, aveD);backwards(sigma_E, disD)

        // update
        updates(myu_E);updates(sigma_E);updates(decoder)
        if(ln-1 == i){
          trainList ::=  y.sum
          train_inList ::= x.sum
        }
      }

      //test
      var testE = List[Double]()
      var testD = List[Double]()
      for(x <- testin){
        // forwards
        val ave = forwards(myu_E,x)
        val dis = forwards(sigma_E,x)
        val z = dis.map(_*rand.nextGaussian).zip(ave).map{case (a,b) => a + b}
        val y = forwards(decoder,z)

        // error
        eError = mse(y,x)
        dError = divergence(ave, dis)
        testE ::= eError
        testD ::= dError

        // resets
        resets(myu_E); resets(sigma_E);resets(decoder)
        if(ln-1 == i){
          testList ::= y.sum
          test_inList ::= x.sum
        }
      }

      var st1 = "down";var st2 = "down";var st3 = "down";var st4 = "down"

      var time = System.currentTimeMillis - start_a
      var e1 = E_train.sum/trainin.size;var e2 = D_train.sum/trainin.size;
      var e3 = testE.sum/trainin.size; var e4 = testD.sum/trainin.size;
     
      if(e1-err1_b > 0){ st1 = "up" }
      if(e2-err2_b > 0){ st2 = "up" }
      if(e3-err3_b > 0){ st3 = "up" }
      if(e4-err4_b > 0){ st4 = "up" }
 
      //print
      println("result: "+i+" time:"+time)
      print("train error E: " +E_train.sum/trainin.size+" !"+st1+"!")
      println(" /train error D:"+D_train.sum/trainin.size+" !"+st2+"!")
      print("test error E: " + testE.sum/testin.size+" !"+st3+"!")
      println(" /test error D:" + testD.sum/testin.size+" !"+st4+"!")

      err1_b = e1
      err2_b = e2
      err3_b = e3
      err4_b = e4
    }

    val pathName = "hist.txt"
    val writer =  new java.io.PrintWriter(pathName)
    val ys1 = trainList.reverse.mkString(",") + "\n"
    val ys2 = testList.reverse.mkString(",") + "\n"
    val ys3 = train_inList.reverse.mkString(",") + "\n"
    val ys4 = test_inList.reverse.mkString(",") + "\n"
    writer.write(ys1)
    writer.write(ys2)
    writer.write(ys3)
    writer.write(ys4) 
    writer.close()

    // run python
    scala.sys.process.Process(
      s"ipython myHist.py $pathName"
    ).run


  }

  def divergence(ave:Array[Double],dis:Array[Double])={
    val ave_sum = ave.map(a => a*a).sum
    val dis_sum = dis.map(a => a*a).sum
    (ave_sum + dis_sum -1 -math.log(dis_sum))/2
  }
  def mse(x:Array[Double],y:Array[Double])={
    x.zip(y).map{case (a,b) =>
      a - b
    }.map(a => a*a).sum
  }

  def createInput(size:Int)={
    val nums =
      for(i <- 0 until size) yield
        Array(rand.nextGaussian*0.1)
    nums.toArray
  }

  def forwards(layers:List[Layer],x:Array[Double])={
    var temp = x
    for(lay <- layers){temp =lay.forward(temp) }
    temp
  }

  def backwards(layers:List[Layer],x:Array[Double])={
    var d = x
    for(lay <- layers.reverse){d = lay.backward(d)}
    d
  }

  def updates(layers:List[Layer])={
    for(lay <- layers){lay.update()}
  }

  def resets(layers:List[Layer]){
    for(lay <- layers){lay.reset()}
  }



}


abstract class Layer {
  def forward(x:Array[Double]) : Array[Double]
  def backward(x:Array[Double]) : Array[Double]
  def update() : Unit
  def reset() : Unit
}



class Affine(val xn:Int, val yn:Int) extends Layer{
  val rand = new scala.util.Random(0)
  var W = DenseMatrix.zeros[Double](yn,xn).map(_ => rand.nextGaussian*0.01)
  for(i <- 0 until yn;j <- 0 until xn){
    W(i,j)=rand.nextGaussian*0.01
  }
  var b = DenseVector.zeros[Double](yn)
  var dW = DenseMatrix.zeros[Double](yn,xn)
  var db = DenseVector.zeros[Double](yn)
  var xs = List[Array[Double]]()
  var t=0
  def push(x:Array[Double]) = { xs ::= x; x }
  def pop() = { val x = xs.head; xs = xs.tail; x }

  def forward(x:Array[Double]) = {
    push(x)
    val xv = DenseVector(x)
    val y = W * xv + b
    y.toArray
  }

  def backward(d:Array[Double]) = {
    val x = pop()
    val dv = DenseVector(d)
    val X = DenseVector(x)
    // dW,dbを計算する ★
    dW += dv * X.t
    db += dv
    var dx = DenseVector.zeros[Double](xn)
    // dxを計算する ★
    dx = W.t * dv
    dx.toArray
  }
  var rt1=1d
  var rt2=1d
  var sW = DenseMatrix.zeros[Double](yn,xn)
  var rW = DenseMatrix.zeros[Double](yn,xn)
  var sb =  DenseVector.zeros[Double](yn)
  var rb =  DenseVector.zeros[Double](yn)

  def update() {
    // W,bを更新する ★
    val epsilon = 0.001
    val rho1=0.9
    val rho2=0.999
    val delta=0.000000001
    var d_tW =DenseMatrix.zeros[Double](yn,xn)
   
    var s_hW = DenseMatrix.zeros[Double](yn,xn)
    var r_hW = DenseMatrix.zeros[Double](yn,xn)

    var d_tb = DenseVector.zeros[Double](yn)
    var s_hb =  DenseVector.zeros[Double](yn)
    var r_hb =  DenseVector.zeros[Double](yn)

    rt1=rt1*rho1
    rt2=rt2*rho2
    t=t+1

   
    for(i <- 0 until yn){
      sb(i) = rho1*sb(i)+ (1 - rho1)*db(i)
      rb(i) = rho2*rb(i) + (1 - rho2)*db(i)*db(i)
      s_hb(i) = sb(i)/(1-rt1)
      r_hb(i) = rb(i)/(1-rt2)
      d_tb(i) = - epsilon * (s_hb(i)/(Math.sqrt(r_hb(i))+delta))
      b(i) = b(i) + d_tb(i)
      for(j <- 0 until xn){
        sW(i,j) =  rho1*sW(i,j) + (1 - rho1)*dW(i,j)
        rW(i,j) =  rho2*rW(i,j) + (1 - rho2)*dW(i,j)*dW(i,j)
        s_hW(i,j) = sW(i,j)/(1-rt1)
        r_hW(i,j) = rW(i,j)/(1-rt2)
        d_tW(i,j) = - epsilon * (s_hW(i,j) /(Math.sqrt(r_hW(i,j))+delta))
        W(i,j) = W(i,j) + d_tW(i,j)
      }
    }
       reset()
  }
  def update_sgd(){
    val lr=0.01
    W -= lr * dW
    b -= lr * db
    reset()
  }
  def reset() {
    dW = DenseMatrix.zeros[Double](yn,xn)
    db = DenseVector.zeros[Double](yn)
    xs = List[Array[Double]]()
  }
}

class Tanh() extends Layer{
  var ys = List[DenseVector[Double]]()
  def tanh(x:Double) = (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
  def forward(xx:Array[Double]) = {
    val x = DenseVector(xx)
    ys ::= x.map(tanh)
    ys.head.toArray
  }

  def backward(d:Array[Double]) = {
    val y = ys.head
    ys = ys.tail
    val ds =DenseVector(d)
    val r = ds *:* (1d - y*y)
    r.toArray
  }

  def update() {
    reset()
  }

  def reset() {
    ys = List[DenseVector[Double]]()
  }

}

class Sigmoid() extends Layer{
  var ys = List[DenseVector[Double]]()
  def sigmoid(x:Double) = 1 / (1 + math.exp(-x))
  def forward(xx:Array[Double]) = {
    val x = DenseVector(xx)
    ys ::= x.map(sigmoid)
    ys.head.toArray
  }

  def backward(d:Array[Double]) = {
    val ds = DenseVector(d)

    val y = ys.head
    ys = ys.tail
    val r =ds *:* y *:* (1d - y)
  
    r.toArray
  }
  def update()={
    reset()
  }
  def reset()={
    ys = List[DenseVector[Double]]()
  }

}
