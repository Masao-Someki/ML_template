# ML_template
機械学習のコードを書くときに用いるテンプレート。前処理とモデル定義さえ書けば、あとはすべて同じコードで動かせるようにしたい

## ディレクトリ構造
```
ML_template
├── LICENSE
├── README.md
├── data                  <- 学習用データを入れるところ
├── log                   <- 学習ログが入るところ
├── model                 <- 学習済みモデルが入るところ
├── path.sh               <- パスを通す
├── run.sh                <- 実行ファイル
├── src                   <- スクリプト
│   └── __init__.py
└── tools
    ├── Makefile
    ├── commands          <- parse_options.sh等の便利ツールを入れておくところ
    └── requirements.txt  <- 必要なパッケージをいれる
```
