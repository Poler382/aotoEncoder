import breeze.linalg._

object layer_util{
  def decoder_choice(mode2:String)={
    var Decoder = mode2 match{
      case "A" =>{
        val a = new Affine(500,32*32*3)
        List(a)
      }
      case "RA" =>{
        val a = new Affine(500,512)
        val b = new Affine(512,32*32*3)
        List(a,b)
      }
    }
    Decoder
  }


  def encoder_choice(mode:String)={
    var Encoder = mode match{
      case "AR" =>{
        val a = new Affine(32*32*3,500)
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
        val i = new Affine(10*5*5,500)
        List(a,b,c,d,e,f,g,h,i)
      }
      case "crpcrpa" => {
        val a = new Convolution(3,32,32,10,5)
        val b = new ReLU()
        val c = new Pooling(2,10,28,28)
        val d = new Convolution(10,14,14,10,5)
        val e = new ReLU()
        val f = new Pooling(2,10,10,10)
        val g = new Affine(250,500)
        List(a,b,c,d,e,f,g)
      }
    }
    Encoder
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

   def saves(layers:List[Layer],fn:String){
    for(lay <- layers){
      lay.save(fn)
    }
  }

  def loads(layers:List[Layer],fn:String){
    for(lay <- layers){
      lay.load(fn)
    }
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

  def make_image2(ys:Array[Array[Double]], NW:Int, NH:Int, H:Int, W:Int) = {
    val im = Array.ofDim[Int](NH * H, NW * W, 3)
    val ymax = ys.flatten.max
    val ymin = ys.flatten.min
    def f(a:Double) = ((a - ymin) / (ymax - ymin) * 255).toInt
    for(i <- 0 until NH; j <- 0 until NW) {
      for(p <- 0 until H; q <- 0 until W; k <- 0 until 3) {
        im(i * H + p)(j * W + q)(k) = f(ys(i * NW + j)(k * H * W + p * W + q))
      }
    }
    im
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


}

