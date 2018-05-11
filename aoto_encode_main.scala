package auto

import java.io.{FileOutputStream=>FileStream,OutputStreamWriter=>StreamWriter}
import breeze.linalg._
import math._


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

class Affine(val xn:Int, val yn:Int) extends Layer{
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

class Pooling(val BW:Int, val IC:Int, val IH:Int, val IW:Int) extends Layer{
  val OH = IH / BW
  val OW = IW / BW
  val OC = IC
  var masks = List[Array[Double]]()
  def push(x:Array[Double]) = { masks ::= x; x }
  def pop() = { val mask = masks.head; masks = masks.tail; mask }

  def iindex(i:Int, j:Int, k:Int) = i * IH * IW + j * IW + k
  def oindex(i:Int, j:Int, k:Int) = i * OH * OW + j * OW + k
  
  def forward(X:Array[Double]) = {
    val mask = push(Array.ofDim[Double](IC * IH * IW))
    val Z = Array.ofDim[Double](OC * OH * OW)
    for(i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      var v = Double.NegativeInfinity
      var row_max = -1
      var col_max = -1
      for(m <- 0 until BW; n <- 0 until BW if v < X(iindex(i,j*BW+m,k*BW+n))) {
        row_max = j*BW+m
        col_max = k*BW+n
        v = X(iindex(i,j*BW+m,k*BW+n))
      }
      mask(iindex(i,row_max,col_max)) = 1
      Z(oindex(i,j,k)) = v
    }
    Z
  }

  def backward(d:Array[Double]) = {
    val mask = pop()
    val D = Array.ofDim[Double](mask.size)
    for(i <- 0 until OC; j <- 0 until OH; k <- 0 until OW) {
      for(m <- 0 until BW; n <- 0 until BW if mask(iindex(i,j*BW+m,k*BW+n)) == 1) {
        D(iindex(i,j*BW+m,k*BW+n)) = d(oindex(i,j,k))
      }
    }
    D
  }

  def update() {
    reset()
  }

  def reset() {
    masks = List[Array[Double]]()
  }
}


class Convolution(
  val I:Int, //入力チャネル数
  val H:Int, //入力の高さ
  val W:Int, //入力の幅
  val O:Int, //出力チャネル数
  val kw:Int //カーネルの幅
) extends Layer {
  val rand = new scala.util.Random(0)
  var K=Array.ofDim[Double](O,I,kw*kw).map(_.map(_.map(a => rand.nextGaussian*0.01)))
  var t=0
  var V2=Array[Double]()
  val w_d=W-kw+1
  val h_d=H-kw+1
  var d_k=Array.ofDim[Double](O,I,kw*kw)

  def vind(i:Int,j:Int,k:Int)=i*H*W+j*W+k
  def zind(i:Int,j:Int,k:Int)=i*(H-kw+1)*(W-kw+1)+j*(W-kw+1)+k

  def forward(V:Array[Double])={
    V2=V
    val Z=Array.ofDim[Double](O*h_d*w_d)
    for(i <- 0 until O ; j <- 0 until h_d; k <- 0 until w_d){
      var s=0d
      for(l <- 0 until I ; m <- 0 until kw ; n <- 0 until  kw){
        s += V(vind(l,j+m,k+n))*K(i)(l)(m*kw+n)
      }
      Z(zind(i,j,k))=s
    }
    Z
  }

  def backward(G:Array[Double])={
    var d_v=Array.ofDim[Double](I*H*W)
    for(i <- 0 until O ;j <- 0 until I ;k <- 0 until kw;l <- 0 until kw){
      var s_k=0d
      for(m <- 0 until h_d; n <- 0 until w_d){
        s_k += G(zind(i,m,n))*V2(vind(j,m+k,n+l))
      }
      d_k(i)(j)(k * kw + l)=s_k
    }
    for(i <- 0 until I;j <- 0 until H;k <- 0 until W){
      var s_v=0d
      for(l <- 0 until h_d;m <- 0 until kw if l+m == j){
        for(n <- 0 until w_d;p <- 0 until kw if n+p ==k){
          for(q <- 0 until O){
            s_v += K(q)(i)(m*kw+p)*G(zind(q,l,n))
          }
        }
        d_v(vind(i,j,k))=s_v
      }
    }
    d_v
  }
  var rt1=1d
  var rt2=1d
  var s=Array.ofDim[Double](O,I,kw*kw)
  var r=Array.ofDim[Double](O,I,kw*kw)
  
  def update()={
    val epsilon = 0.001
    val rho1=0.9
    val rho2=0.999
    val delta=0.000000001
    var d_t=Array.ofDim[Double](O,I,kw*kw)
    var s_h=Array.ofDim[Double](O,I,kw*kw)
    var r_h=Array.ofDim[Double](O,I,kw*kw)
    rt1=rt1*rho1
    rt2=rt2*rho2
    t=t+1
    for(i <- 0 until O; j <- 0 until I; k <- 0 until kw*kw){
      s(i)(j)(k) = rho1*s(i)(j)(k) + (1 - rho1)*d_k(i)(j)(k)
      r(i)(j)(k) = rho2*r(i)(j)(k) + (1 - rho2)*d_k(i)(j)(k)*d_k(i)(j)(k)
      s_h(i)(j)(k) = s(i)(j)(k)/(1-rt1)
      r_h(i)(j)(k) = r(i)(j)(k)/(1-rt2)
      d_t(i)(j)(k) = - epsilon * (s_h(i)(j)(k)/(math.sqrt(r_h(i)(j)(k))+delta))
      K(i)(j)(k) = K(i)(j)(k) + d_t(i)(j)(k)
    }
    reset()
  }
  def reset() {
    d_k=Array.ofDim[Double](O,I,kw*kw)
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

