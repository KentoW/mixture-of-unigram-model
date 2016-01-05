# mixture-of-unigram-model
##概要
混合ユニグラムモデル(mixture of unigram model)をPythonで実装  
無限混合ユニグラムモデル(infinite mixture of unigram model)をPythonで実装  
[Wiki.py](https://github.com/KentoW/wiki)を使用して
##mixture_of_unigram_model.pyの使い方(混合ユニグラムモデル)
```python
# Sample code.
from mixture_of_unigram_model import MUM

alpha = 0.01    # 初期ハイパーパラメータalpha
beta = 0.01     # 初期ハイパーパラメータbeta
K = 10          # トピック数
N = 1000        # 最大イテレーション回数
converge = 0.01 # イテレーション10回ごとに対数尤度を計算し，その差分(converge)が小さければ学習を終了する

mum = MUM()
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

imum = IMUM()
imum.set_param(alpha, beta, N, converge)
imum.learn()
imum.output_model()
```
