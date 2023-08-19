"""
lightgbm
    lightgbmはboostingだから、baggingと違って、バイアスを改善する効果があるがバリアンスを改善しない。よって過学習しないようにパラメータを気を付ける必要がある。
    そこが精度が低い原因か？他の学習手法と一緒に平均（重み付き）すれば改善するのか。
    多数決の場合は予測が真値と離れてても少しなら影響ないが、平均なら絶対影響あるから、
    他の学習手法と合わせて平均することに甘んじずに、一つ一つの弱識別器の過学習に気をつけるべき

特徴量の作成(Feature Engineering)
    年間総距離は走行距離を年数で割る
    欠損値があるかとか、instance欠損値、、数とか〇
    Umapして次元削減したり、Umapの結果をfeatureに加えたり
    denoising autoencoderで、次元削減したり、新たなfeatureに加えたり
    クラスタリングでinstanceごとに、クラスターまでの距離をfeatureに加えたり

前処理 
    odometerのデータの違和感の調査 走行距離がマイナス・極端に大きい数値に何かあるか確認し、必要があれば修正
    region：embeddingやる
    manufacturer〇
    cylinders〇
    size〇
    state,type,paint_color,manufacturerをtarget encodingやる〇
    train_dataの平均とかを使って、testデータを標準化〇※上手くいかなかった、trainとtestを縦に結合してms.fitしたやつを使って、それぞれtransformしてもいいかも
    数値データをrankgaussやるとか


    feature(resion)をembeddingして、それを次元削減したい
"""