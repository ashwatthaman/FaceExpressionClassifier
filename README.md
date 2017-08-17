# FaceExpressionClassifier

台詞を入力すると、感情判定を行うコードです。
解説は以下の記事にあります。
http://qiita.com/thetenthart/items/04b220ea8d348ccdaed6

./src/predict.pyの最下部のtext_arrを書き換えると、好きな台詞を判定できます。

実行にはsentencepieceが必要です。
https://github.com/google/sentencepiece

他に必要なライブラリはnumpy, chainerなどです。
chainerはver2以降で動きます。
