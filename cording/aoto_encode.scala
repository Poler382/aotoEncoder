package auto
import java.io.{FileOutputStream => FileStream, OutputStreamWriter => StreamWriter}

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
      load_data(dir+"/train1-d.txt",dir+"/train1-t.txt")
    val (test_d,test_t) =
      load_data(dir+"/test1-d.txt", dir+"/test1-t.txt")

    println("load finish")

    (train_d.map(sortColor(_)).zip(train_t), (test_d.map(sortColor(_))).zip(test_t))
  }

  def sub(a:Array[Double],b:Array[Double])={
    var sub = Array.ofDim[Double](a.size)
    for(i <- 0 until a.size){
      sub(i) = a(i) - b(i)
    }
    sub
  }

  def add_noise(x:Array[Double],flag:Int)={
    var ds = new Array[Double](x.size)

    for(i <- 0 until x.size){
      ds(i) = x(i)
    }

    if(flag == 1){
      var stop = (rand.nextInt(3)+12) * 100
      for(i <- 0 until stop ){
        ds(rand.nextInt(32*32*3)) = 0
      }
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

  def plotdata(fn:String,state:String,ylab:String,trainList:List[Double],testList:List[Double]){
      
    val pathName = fn+".txt"
    val writer =  new java.io.PrintWriter(pathName)
    val ys1 = trainList.reverse.mkString(",") + "\n"
    val ys2 = testList.reverse.mkString(",") + "\n"
    val title = state+"\n"
    val ylabel = ylab+"\n"
    writer.write(title)
    writer.write(ylabel)
    writer.write(ys1)
    writer.write(ys2)
    writer.close()

    /* run python
    scala.sys.process.Process(
      s"ipython myPlot.py $pathName"
    ).run
     */

  }
  val e_mode = Array("AR")
  val d_mode = Array("A")
  val noize_mode = Array(0,1)
  def main(args:Array[String]){
    
    // データの読み込み
    ///home/share/cifar10
     val (dtrain,dtest) = load_cifer("C:/Users/poler/Documents/python/share")

    for (one <- e_mode; two <- d_mode; three <- noize_mode){

      val mode = one
      val mode2= two
      val noise = three
      val ln   = args(0).toInt // 学習回数       ★
      val dn   = args(1).toInt // 学習データ数    ★
      val tn   = args(2).toInt // テストデータ数  ★

      val layermode = mode+mode2
      val state = ln+"_"+dn+"_"+tn
      var line = "mode_"+mode+mode2+"_"+"ln_"+ln+"-"+"dn_"+dn+"_"+"tn_"+tn+"_noise_"+noise.toString
      var Encoder = layer_util.encoder_choice(mode)
      var Decoder = layer_util.decoder_choice(mode2)
      println("\n"+line)
      var c = new Affine(500,10)//分類用

      val layers = Encoder ++ Decoder
      val c_layers= (c::(Encoder.reverse)).reverse

      //aotoEncoder learning
      var num = 0

      var MS_train1 = List[Double]()
      var MS_test1  = List[Double]()
      var MS_train2 = List[Double]()
      var MS_test2  = List[Double]()
      var AC_train  = List[Double]()
      var AC_test   = List[Double]()

      val total_time = System.currentTimeMillis
      for ( i <- 0 until ln){
        num = 0

        var err1 = 0d; var err2 = 0d
        var start_l = System.currentTimeMillis
        var ys=List[Array[Double]]()
        for((x,n) <- dtrain.take(dn) ) {
          val xtemp = x

          var y = layer_util.forwards(layers,add_noise(xtemp,noise))

          layer_util.backwards(layers,sub(y,x))

          ys ::= y

          if(rand.nextInt(10) == 1){//min-butch
            layer_util.updates(layers)
          }
          ys = ys.reverse 
          err1 += sub(y,x).map(a => a*a).sum
        }
        if(i == ln-1 || i % 500 == 0){
          Image.write("train_ln"+i.toString+"_"+line+".png",Image.make_image2(ys.toArray,10,10,32,32))
        }
        num = 0
        var c_count = 0d
        var as=List[Array[Double]]()
        for((x,n) <- dtest.take(tn) ){
          val xtemp = x
          val y = layer_util.forwards(layers,add_noise(xtemp,noise))
          val d = sub(y,x)
          err2 += sub(y,x).map(a => a*a).sum
          as ::= y
          
        }
        as = as.reverse  
        if(i == ln-1 || i % 500 == 0){
          Image.write("test_ln"+i.toString+"_"+line+".png",Image.make_image2(as.toArray,10,10,32,32))
        }
        var time = System.currentTimeMillis - start_l
        MS_test1 ::= err2/tn
        MS_train1 ::= err1/dn
        learning.print_result(i,time,err1/dn,err2/tn,0,0,0,0,dn,tn)
      }


      println("\nrecognaize")
      ///認識

      for(i <- 0 until ln){
        var err1 = 0d
        var err2 = 0d
        var a_count = 0d
        var dropnum = 100
        var start_a =System.currentTimeMillis
        for((x,n) <- dtrain.take(dropnum) ) {
          var y1 = layer_util.forwards(c_layers,add_noise(x,noise))
          layer_util.backwards(c_layers,sub(y1,onehot(n)))
          if(rand.nextInt(10)==1){
            layer_util.updates(c_layers)
          }

          err1 += sub(y1,x).map(a => a*a).sum
          if(argmax(y1) == n){
            a_count+=1
          }
        }

        var c_count = 0d
        for((x,n) <- dtest.take(dropnum) ){
          val yy = layer_util.forwards(c_layers,add_noise(x,noise))
          err2 += sub(yy,x).map(a => a*a).sum
          if(argmax(yy) == n){ c_count+=1 }

        }

        var time = System.currentTimeMillis - start_a
        learning.print_result(i,time,err1/dn,err2/tn,0,0,a_count,c_count,dropnum,dropnum)
        MS_test2 ::= err2/tn
        MS_train2 ::= err1/dn
        AC_test ::= a_count/dropnum * 100
        AC_train ::= c_count/dropnum * 100
      }

      println("\ntotal time: "+(System.currentTimeMillis-total_time)/1000d)
      plotdata("try1_"+line,line+"data1","Mean Square Error",MS_train1,MS_test1)
      plotdata("try2_"+line,line+"data2_1","Mean Square Error",MS_train2,MS_test2)
      plotdata("try3_"+line,line+"data2_2","Accuary Rate",AC_train,AC_test)

    }

  }
}
abstract class Layer {
  def forward(x:Array[Double]) : Array[Double]
  def backward(x:Array[Double]) : Array[Double]
  def update() : Unit
  def reset() : Unit
}
