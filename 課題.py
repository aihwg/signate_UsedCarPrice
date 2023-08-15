"""
lightgbm
    lightgbmはboostingだから、baggingと違って、バイアスを改善する効果があるがバリアンスを改善しない。よって過学習しないようにパラメータを気を付ける必要がある。
    そこが精度が低い原因か？他の学習手法と一緒に平均（重み付き）すれば改善するのか。
    多数決の場合は予測が真値と離れてても少しなら影響ないが、平均なら絶対影響あるから、
    他の学習手法と合わせて平均することに甘んじずに、一つ一つの弱識別器の過学習に気をつけるべき

SVM,RF,NNあたりでスタッキング(各モデルごとに用いるfeatureを変えたりとかして、モデル間の相関を減らす)

特徴量の作成(Feature Engineering)
    年間総距離は走行距離を年数で割る
    欠損値があるかとか、欠損値の数とか、そのパターンとか

前処理 
odometerのデータの違和感の調査 走行距離がマイナス・極端に大きい数値に何かあるか確認し、必要があれば修正
region：多すぎ、ダミー変数で対応？
manufacturer〇
cylinders
drive
size

標準化を、train_dataの平均分散でtestdataも行う(二値変数は標準化しない)
"""