"""
lightgbmm
    lightgbmはboostingだから、baggingと違って、バイアスを改善する効果があるがバリアンスを改善しない。よって過学習しないようにパラメータを気を付ける必要がある。
    そこが精度が低い原因か？他の学習手法と一緒に平均（重み付き）すれば改善するのか。
    多数決の場合は予測が真値と離れてても少しなら影響ないが、平均なら絶対影響あるから、
    他の学習手法と合わせて平均することに甘んじずに、一つ一つの弱識別器の過学習に気をつけるべき

特徴量の作成(Feature Engineering)
    年間総距離は走行距離を年数で割る、販売する時期を2023とする〇
    欠損値があるかとか、instance欠損値、、数とか〇
    〇Umapして次元削減したり、Umapの結果をfeatureに加えたり
    〇denoising autoencoderで、次元削減したり、新たなfeatureに加えたり
    〇クラスタリングでinstanceごとに、クラスターまでの距離をfeatureに加えたり

前処理 
    odometerのデータの違和感の調査 走行距離がマイナス・極端に大きい数値に何かあるか確認し、必要があれば修正i〇
    year〇
    region：embeddingやる〇※label encodingの方が良かった
    region：地域ごとの緯度と経度を説明変数に
    condition：数値データに直す（またはenbedding）〇
    paintcolor:ラベルデータでいいかもm〇そうでもなかったかも
    manufacturer〇
    cylinders〇
    size〇
    state,type,paint_color,manufacturerをtarget encodingやる〇
    標準化：train_dataの平均とかを使って、testデータを標準化〇※上手くいかなかった
    〇標準化：trainとtestをconcatしてから、標準化してみる
    〇数値データをrankgaussやるとか
    
    
    前処理year 2023以上を-1000
    前処理manufacturer 全角半角変換
    前処理condition　カテゴリ変数から数値変数
    前処理cylinders　カテゴリ変数から数値変数(other=-1)
    前処理size　カテゴリ変数から数値変数
    前処理state,type,paint_color,manufacturer　target encoding
    特徴量追加　行ごとに欠損値あるか
    ---------------------------------------------
    train error : 0.6069744648573043
    valid error : 0.6582146842045825

    前処理odometer -131869を131869にした
    前処理year 2023以上を-1000
    前処理manufacturer 全角半角変換
    前処理condition　カテゴリ変数から数値変数
    前処理cylinders　カテゴリ変数から数値変数(other=-1)
    前処理size　カテゴリ変数から数値変数
    前処理state,type,paint_color,manufacturer　target encoding
    特徴量追加　行ごとに欠損値あるか
    ---------------------------------------------
    train error : 0.6085665612355184
    valid error : 0.6583049508135316

    前処理year 2023以上を-1000
    Feature Engineering：年間走行距離
    前処理manufacturer 全角半角変換
    前処理condition　カテゴリ変数から数値変数
    前処理cylinders　カテゴリ変数から数値変数(other=-1)
    前処理size　カテゴリ変数から数値変数
    前処理state,type,paint_color,manufacturer　target encoding
    特徴量追加　行ごとに欠損値あるか
    ---------------------------------------------
    train error : 0.6047578769248435
    valid error : 0.6582718075213234
    
"""