package auto



object learning{
  val rand = new scala.util.Random(0)





  //学習回数　かかった時間　誤差１…4　
  //学習時正解データ数　テスト時正解データ数 学習データ数　テストデータ数
  def print_result(
    num:Int,
    time:Double,
    err1:Double,
    err2:Double,
    err3:Double,
    err4:Double,
    countL:Double,
    countT:Double,
    dn:Int,
    tn:Int){
    var printdata = "result:"+num.toString

    if(time != 0d){
      printdata+= " / time: "+(time/1000d).toString+"\n"
    }
    if(err1 != 0d){
      printdata += " err1: "+err1.toString +" /"
    }
    if(err2 != 0d){
      printdata += " err2: "+err2.toString +" /"
    }
    if(err3 != 0d){
      printdata += " err3: "+err3.toString +" /"
    }
    if(err4 != 0d){
      printdata += " err4: "+err4.toString +" /"
    }

    printdata += "\n"

    if(countL != 0d){
      printdata += " /learning rate: " + (countL/dn * 100).toString
    }

    if(countT != 0d){
      printdata += " /learning rate: " + (countT/tn * 100).toString
      printdata += "\n"
    }

    

    print(printdata)

  }


}
