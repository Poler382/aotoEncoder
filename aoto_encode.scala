package auto
import breeze.linalg._
import math._
import java.io.{FileOutputStream=>FileStream,OutputStreamWriter=>StreamWriter}

object AE{

  def load_mnist(dir:String) = {
    def fd(line:String) = line.split(",").map(_.toDouble / 256).toArray
    def ft(line:String) = line.split(",").map(_.toInt).toArray
    var train_d = scala.io.Source.fromFile(dir + "/train-d.txt").getLines.map(fd).toArray
    val train_t = scala.io.Source.fromFile(dir + "/train-t.txt").getLines.map(ft).toArray.head
    var test_d = scala.io.Source.fromFile(dir + "/test-d.txt").getLines.map(fd).toArray
    val test_t = scala.io.Source.fromFile(dir + "/test-t.txt").getLines.map(ft).toArray.head
    (train_d.zip(train_t), test_d.zip(test_t))
  }

  def sub(a:Array[Double],b:Array[Double])={
    var sub = Array.ofDim[Double](a.size)
    for(i <- 0 until a.size){
      sub(i) = a(i) - b(i)
    }
    sub
  }

  def forwards(layers:List[Layer],x:Array[Double])={
    var temp = x
    for(lay <- layers){
      temp =lay.forward(temp)
    }
    temp
  }

  def backwards(layers:List[Layer],x:Array[Double])={
    var d = x
    for(lay <- layers.reverse){
      d = lay.backward(d)
    }
    d
  }

  def updates(layers:List[Layer])={
    for(lay <- layers){
      lay.update()
    }
  }

  def resets(layers:List[Layer]){
    for(lay <- layers){
      lay.reset()
    }
  }


  def writer_file(fn:String,ln:String)={

    val out_put1 =fn+".txt"
    val encode = "UTF-8"
    val append = false

    val fileOutPutStream1 = new FileStream(out_put1, append)
    val writer1 = new StreamWriter( fileOutPutStream1, encode)

    writer1.write(ln.toString)
    println(out_put1+"：書き込み完了")

    writer1.close
  }
  var num = 0
  def decode_rgb(rgb:Array[Double])={
    var rhead = 0
    var ghead = 32*32
    var bhead = 32*32*2

    var image = Array.ofDim[Int](32,32,3)

    for (i <- 0 until 32;j <- 0 until 32){
    // print(rgb(i+j)+" -> ")
      image(i)(j)(0) = (rgb(rhead+i+j)*256).toInt
      image(i)(j)(1) = (rgb(ghead+i+j)*256).toInt
      image(i)(j)(2) = (rgb(bhead+i+j)*256).toInt
    //  println(image(i)(j)(0))
    }

    for(i <- 0 until 32;j <- 0 until 32; rgb <- 0 until 3 ){
    //  println(image(i)(j)(rgb))
    }

    Image.write("ttt"+num.toString+".png",image)
    num += 1
    image

  }

  def makeimg(ts:Array[Array[Array[Double]]],as:Array[Array[Array[Double]]])={
    val size = 32
    val return_file = Array.ofDim(32,32,3)
    println(ts.size)
    println(ts(0).size)
    println(ts(0)(0).size)


    for(i <- 0 until size){
      for(j <- 0 until size){
        for(rgb <- 0 until 3){
          println(ts(i)(j)(rgb))
        }
      }
    }
  }


  def makergb(rgb:Array[Double])={
    var RGB = new Array[Double](32*32*3)
    var j =0
    for(i <- 0 until rgb.size){
      var num = 32*32*(i % 3)+j
      RGB(num) = rgb(i)
      if(i%3 == 2){
        j+=1
      }
    }
    RGB
  }

  def main(args:Array[String]){
    /*
     val mode = args(0)
     val ln   = args(1).toInt // 学習回数       ★
     val dn   = args(2).toInt // 学習データ数    ★
     val tn   = args(3).toInt // テストデータ数  ★
     println("ln:"+ln+"/"+"dn:"+dn+"/"+"tn:"+tn)
     */

    val mode = "basic"
    val ln   = 20 // 学習回数       ★
    val dn   = 100 // 学習データ数    ★
    val tn   = 100 // テストデータ数  ★

   

    val layers = mode match{
      case "basic" =>{
        val a = new Affine(32*32*3,128)
        val b = new ReLU()
        val c = new Affine(128,32*32*3)
        
        List(a,b,c)
      }
    }



    val rand = new scala.util.Random(0)

    // データの読み込み
    val (dtrain,dtest) = load_mnist("/home/share/cifar10")
    println("finish read\n")
    //学習
    val ds = (0 until 100).map(i => makergb(dtrain(i)._1)).toArray
    var err1_b = 0d
    var err2_b = 0d

    for ( i <- 0 until ln){
      var err1 = 0d
      var err2 = 0d   
      var start_l = System.currentTimeMillis
      // for((x,n) <- rand.shuffle(dtrain.toList).take(dn) ) {
      var x = Array.ofDim[Double](32*32*3)
      var y = Array.ofDim[Double](32*32*3)

      for((x1,n) <- dtrain.take(dn) ) {
        x = makergb (x1)
        y = forwards(layers,x)
        val d = sub(y,x)
        backwards(layers,sub(y,x))
        updates(layers)
        err1 += sub(y,x).map(a => a*a).sum
      }

      val ll = decode_rgb(y)

      for((x1,n) <- dtest.take(tn) ){
        var x = makergb (x1)
        val y = forwards(layers,x1)
        val d = sub(y,x)
        err2 += sub(y,x).map(a => a*a).sum

    //    val ll = decode_rgb(x)
 
   //     val im = makeimg(decode_rgb(x),decode_rgb(y) )


      }

      var st1 = "down"
      var st2 = "down"
      var time = System.currentTimeMillis - start_l


      if(err1-err1_b > 0){ st1 = "up" }
      if(err2-err2_b > 0){ st2 = "up" }
      print("learning :"+i+"/time: "+time/1000d)
      print(" /err1: "+err1/dn+" -"+st1)
      println("/err2: "+err2/tn+" -"+st2)
      err1_b = err1
      err2_b = err2
    }
  }
}

abstract class Layer {
  def forward(x:Array[Double]) : Array[Double]
  def backward(x:Array[Double]) : Array[Double]
  def update() : Unit
  def reset() : Unit
}

class ReLU() extends Layer {
  var ys = List[Array[Double]]()
  def push(y:Array[Double]) = { ys ::= y; y }
  def pop() = { val y = ys.head; ys = ys.tail; y }

  def forward(x:Array[Double]) = {
    push(x.map(a => math.max(a,0)))
  }

  def backward(d:Array[Double]) = {
    val y = pop()
      (0 until d.size).map(i => if(y(i) > 0) d(i) else 0d).toArray
  }

  def update() {
    reset()
  }

  def reset() {
    ys = List[Array[Double]]()
  }
}

class Affine(val xn:Int, val yn:Int) extends Layer {
  val rand = new scala.util.Random(0)
  var W = DenseMatrix.zeros[Double](yn,xn)//.map(_ => rand.nextGaussian*0.01)
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

object Image {
  def rgb(im : java.awt.image.BufferedImage, i:Int, j:Int) = {
    val c = im.getRGB(i,j)
    Array(c >> 16 & 0xff, c >> 8 & 0xff, c & 0xff)
  }

  def pixel(r:Int, g:Int, b:Int) = {
    val a = 0xff
    ((a & 0xff) << 24) | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff)
  }

  def read(fn:String) = {
    val im = javax.imageio.ImageIO.read(new java.io.File(fn))
    (for(i <- 0 until im.getHeight; j <- 0 until im.getWidth)
    yield rgb(im, j, i)).toArray.grouped(im.getWidth).toArray
  }

  def write(fn:String, b:Array[Array[Array[Int]]]) = {
    val w = b(0).size
    val h = b.size
    val im = new java.awt.image.BufferedImage(w, h, java.awt.image.BufferedImage.TYPE_INT_RGB);
    for(i <- 0 until im.getHeight; j <- 0 until im.getWidth) {
      im.setRGB(j,i,pixel(b(i)(j)(0), b(i)(j)(1), b(i)(j)(2)));
    }
    javax.imageio.ImageIO.write(im, "png", new java.io.File(fn))
  }
}
