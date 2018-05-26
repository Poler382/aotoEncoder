object learning{
  val rand = new scala.util.Random(0)
  //学習回数　かかった時間　誤差１…4　
  //学習時正解データ数　テスト時正解データ数 学習データ数　テストデータ数
  def print_result(
    num:Int,
    time:Double,
    errlist:List[Double],
    countL:Double,
    countT:Double,
    dn:Int,
    tn:Int){
    var printdata = "result:"+num.toString+" - time:"+(time/1000d).toString+"\n"

    for(i <- 0 until errlist.size){
      printdata += "err"+(i+1).toString+":"+errlist(i).toString+"/"
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
