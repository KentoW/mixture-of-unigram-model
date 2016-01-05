# mixture-of-unigram-model
##概要
混合ユニグラムモデル(mixture of unigram model)をPythonで実装  
無限混合ユニグラムモデル(infinite mixture of unigram model)をPythonで実装  
##mixture_of_unigram_model.pyの使い方(混合ユニグラムモデル)
```python
# Sample code.
from mixture_of_unigram_model import MUM

alpha = 0.01    # 初期ハイパーパラメータalpha
beta = 0.01     # 初期ハイパーパラメータbeta
K = 10          # トピック数
N = 1000        # 最大イテレーション回数
converge = 0.01 # イテレーション10回ごとに対数尤度を計算し，その差分(converge)が小さければ学習を終了する

mum = MUM("data.txt")
mum.set_param(alpha, beta, K, N, converge)
mum.learn()
mum.output_model()
```
##infinite_mixture_of_unigram_model.pyの使い方(無限混合ユニグラムモデル)
```python
# Sample code.
from infinite_mixture_of_unigram_model import IMUM

alpha = 0.01    # 初期ハイパーパラメータalpha
beta = 0.01     # 初期ハイパーパラメータbeta
N = 1000        # 最大イテレーション回数
converge = 0.01 # イテレーション10回ごとに対数尤度を計算し，その差分(converge)が小さければ学習を終了する

imum = IMUM("data.txt")
imum.set_param(alpha, beta, N, converge)
imum.learn()
imum.output_model()
```
##入力フォーマット
1単語をスペースで分割した1行1文書形式  
先頭に#(シャープ)記号を入れてコメントを記述可能
```
# 文書1
単語1 単語2 単語3 ...
# 文書2
単語10 単語11 単語11 ...
...
```
例として[Wiki.py](https://github.com/KentoW/wiki)を使用して収集した アニメのあらすじ文章をdata.txtに保存
