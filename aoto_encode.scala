package auto
import breeze.linalg._
import math._
import scala.sys.process.Process
import java.io.{FileOutputStream=>FileStream,OutputStreamWriter=>StreamWriter}

object AE{

  val rand = new scala.util.Random(0)
  def load_mnist(dir:String) = {
    def fd(line:String) = line.split(",").map(_.toDouble / 256).toArray
    def ft(line:String) = line.split(",").map(_.toInt).toArray
    var train_d = scala.io.Source.fromFile(dir + "/train-d.txt").getLines.map(fd).toArray
    val train_t = scala.io.Source.fromFile(dir + "/train-t.txt").getLines.map(ft).toArray.head
    var test_d = scala.io.Source.fromFile(dir + "/test-d.txt").getLines.map(fd).toArray
    val test_t = scala.io.Source.fromFile(dir + "/test-t.txt").getLines.map(ft).toArray.head
    (train_d.zip(train_t), test_d.zip(test_t))
  }

  def sortColor(image:Array[Double]) = {
    var red = List[Double]()
    var green = List[Double]()
    var blue = List[Double]()

    for(i <- 0 until image.size) {
      if(i % 3 == 0) red ::= image(i)
      else if(i % 3 == 1) green ::= image(i)
      else if(i % 3 == 2) blue ::= image(i)
    }

    (red.reverse ++ green.reverse ++ blue.reverse).toArray
  }
  def onehot(a:Int)={
    var t = new Array[Double](10)
    t(a) = 1d
    t
  }

  def argmax(a:Array[Double]) = a.indexOf(a.max)

  def load_data(dpath:String, tpath:String) = {
    def fd(line:String) = line.split(",").map(_.toDouble / 256).toArray
    def ft(line:String) = line.split(",").map(_.toInt).toArray
    val train_d = scala.io.Source.fromFile(dpath).getLines.map(fd).toArray
    val train_t = scala.io.Source.fromFile(tpath).getLines.map(ft).toArray.head
    (train_d, train_t)
  }

  def load_cifer(dir:String) = {
    val (train_d,train_t) =
      load_data(dir+"/train-d.txt",dir+"/train-t.txt")
    val (test_d,test_t) =
      load_data(dir+"/test-d.txt", dir+"/test-t.txt")
    (train_d.map(sortColor(_)).zip(train_t), (test_d.map(sortColor(_))).zip(test_t))
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


  def add_noise(x:Array[Double])={
    var ds = x
    var stop = (rand.nextInt(3)+3) * 100
    for(i <- 0 until stop ){
      ds(rand.nextInt(32*32*3)) = 0
    }
    ds
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

  def makeimg(n:Int,fn:String,f:String)={
    
    for(i <- 0 until 10){
      sys.process.Process("convert +append " 
        +fn+(i*10).toString+  ".png "
        +fn+(i*10+1).toString+".png "
        +fn+(i*10+2).toString+".png "
        +fn+(i*10+3).toString+".png "
        +fn+(i*10+4).toString+".png "
        +fn+(i*10+5).toString+".png "
        +fn+(i*10+6).toString+".png "
        +fn+(i*10+7).toString+".png "
        +fn+(i*10+8).toString+".png "
        +fn+(i*10+9).toString+".png "
        +fn+"file"+i.toString+".png" ).run
      Thread.sleep(300)
    }
  
    sys.process.Process("convert -append "
      +fn+"file0.png "
      +fn+"file1.png "
      +fn+"file2.png "
      +fn+"file3.png " 
      +fn+"file4.png "
      +fn+"file5.png "
      +fn+"file6.png "
      +fn+"file7.png " 
      +fn+"file8.png " 
      +fn+"file9.png " 
      +fn+f+"cifer"+n.toString+".jpg" 
    ).run
  }

  var make_imagelist= List[String]()
  def remove_image(){
    Thread.sleep(300)
     for(i <- make_imagelist){
      if(i.contains(".png") ){
        sys.process.Process("rm "+i).run
      }
    }
    make_imagelist = List()
  }
  def to3DArrayOfColor(image:Array[Double],h:Int,w:Int) = {
    val input = image.map(_*256)
    var output = List[Array[Array[Double]]]()
    for(i <- 0 until h) {
      var row = List[Array[Double]]()
      for(j <- 0 until w) {
        val red = input(i*w+j)
        val green = input(i*w+j+h*w)
        val blue = input(i*w+j+h*w*2)
        row ::= Array(red,green,blue)
      }
      output ::= row.reverse.toArray
    }
    output.reverse.toArray.map(_.map(_.map(_.toInt)))
  }

  def imageWriter(fn:String,im:Array[Array[Array[Int]]])={
    make_imagelist ::= fn
    Image.write(fn,im)
  }

  def main(args:Array[String]){
    
    val mode = args(0)
    val mode2= args(1)
    val ln   = args(2).toInt // 学習回数       ★
    val dn   = args(3).toInt // 学習データ数    ★
    val tn   = args(4).toInt // テストデータ数  ★
    println("ln:"+ln+"/"+"dn:"+dn+"/"+"tn:"+tn)
    val state = ln+"_"+dn+"_"+tn
    var line = "mode:"+mode+mode2+"/"+"ln:"+ln+"/"+"dn:"+dn+"/"+"tn:"+tn+"\n"
    var Encoder = mode match{
      case "AR" =>{
        val a = new Affine(32*32*3,30)
        val b = new ReLU()
        List(a,b)
      }
      case "ccrpccrpa" => {
        val a = new Convolution(3,32,32,10,4)
        val b =  new Convolution(10,29,29,10,4)
        val c = new ReLU()
        val d = new Pooling(2,10,26,26)
        val e = new Convolution(10,13,13,10,3)
        val f = new Convolution(10,11,11,10,2)
        val g = new ReLU()
        val h = new Pooling(2,10,10,10)
        val i = new Affine(10*5*5,30)
        List(a,b,c,d,e,f,g,h,i)
      }
      case "crpcrpa" => {
        val a = new Convolution(3,32,32,10,5)
        val b = new ReLU()
        val c = new Pooling(2,10,28,28)
        val d = new Convolution(10,14,14,10,5)
        val e = new ReLU()
        val f = new Pooling(2,10,10,10)
        val g = new Affine(250,30)
        List(a,b,c,d,e,f,g)
      }
    }

    var Decoder = mode2 match{
      case "A" =>{
        val a = new Affine(30,32*32*3)
        List(a)
      }
      case "RA" =>{
        val a = new Affine(30,512)
        val b = new Affine(512,32*32*3)
        List(a)
      }
    }

    var c = new Affine(30,10)
   
    val layers = Encoder ++ Decoder
    val c_layers= (c::(Encoder.reverse)).reverse

    // データの読み込み
    val (dtrain,dtest) = load_cifer("/home/share/cifar10")
    println("finish read\n")

    //aotoEncoder learning
    var err1_b = 0d;var err2_b = 0d
    var num = 0
    for ( i <- 0 until ln){
      num = 0
      var err1 = 0d;var err2 = 0d   
      var start_l = System.currentTimeMillis
      
      for((x,n) <- dtrain.take(dn) ) {
        var y = forwards(layers,add_noise(x))
        var y1 = forwards(c_layers,add_noise(x))
        
        backwards(layers,sub(y,x))
        backwards(c_layers,sub(y1,onehot(n)))
        if(rand.nextInt(10)==0){
          updates(layers);updates(c_layers)
        }
       
        val returny = to3DArrayOfColor(y,32,32)

        imageWriter("out_"+state+num.toString+".png",returny)
        num+=1
        err1 += sub(y,x).map(a => a*a).sum
      }
            
      makeimg(i,"out_"+state,mode+state)
      remove_image()
      num = 0
      var c_count = 0d
      for((x,n) <- dtest.take(tn) ){

        val y = forwards(layers,add_noise(x))
        val d = sub(y,x)
        err2 += sub(y,x).map(a => a*a).sum
        imageWriter("test_"+state+num.toString+".png", to3DArrayOfColor(y,32,32))
        
        num+=1
        val yy = forwards(c_layers,add_noise(x))
        if(argmax(yy) == n){ c_count+=1  }
      }

      makeimg(i,"test_"+state,mode+state)
      remove_image()

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


    ///認識
    for(i <- 0 until ln){
      var err1 = 0d
      var err2 = 0d
      var a_count = 0d
      var start_a =System.currentTimeMillis
      for((x,n) <- dtrain.take(1000) ) {
        var y1 = forwards(c_layers,add_noise(x))
        backwards(c_layers,sub(y1,onehot(n)))
        if(rand.nextInt(5)==0){
          updates(c_layers)
        }
        
        err1 += sub(y1,x).map(a => a*a).sum
        if(argmax(y1) == n){
          a_count+=1
        }
      }
            
      var c_count = 0d
      for((x,n) <- dtest.take(1000) ){

        val yy = forwards(c_layers,add_noise(x))
        err2 += sub(yy,x).map(a => a*a).sum

        if(argmax(yy) == n){
          c_count+=1
        }

      }

      var st1 = "down"
      var st2 = "down"
      var time = System.currentTimeMillis - start_a

      if(err1-err1_b > 0){ st1 = "up" }
      if(err2-err2_b > 0){ st2 = "up" }
      print("result :"+i+"/time: "+time/1000d)
      print(" /learning rate:" + a_count/1000 * 100)
      print(" /test rate:" + c_count/1000 * 100)
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
