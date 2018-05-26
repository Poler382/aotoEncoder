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

