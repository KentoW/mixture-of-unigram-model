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
##出力フォーマット
必要な情報は各自で抜き取って使用してください．
```
model	mixture_of_unigram_model        # 学習の種類
@parameter
corpus_file	data.txt                    # トレーニングデータのPATH
hyper_parameter_alpha	1.834245        # ハイパーパラメータalpha
hyper_parameter_beta	0.089558        # ハイパーパラメータbeta
number_of_topic	10          # トピック数
number_of_iteration	121     # 収束した時のイテレーション回数
@likelihood         # 尤度
initial likelihood	-1389.55970144
last likelihood	-1382.11395248
@vocaburary         # 学習で使用した単語v
target_word	出産
target_word	拓き
target_word	土
target_word	吉日
target_word	遂げる
...
@count
topic_document_freq	1	109     # トピック分布に必要な情報 左の数字から順に トピックID, そのトピックが割り当てられた文書の数
topic_document_freq	2	167
topic_document_freq	3	52
...
topic_word_sum	1	18137   # 単語生成確率分布に必要な情報 左の数字から順に トピックID, そのトピックが割り当てられた単語の数
topic_word_freq	1	の	1111    # 単語生成確率分布に必要な情報 左の数字から順に トピックID, 単語v, そのトピックが割り当てられた単語vの数
topic_word_freq	1	に	761
topic_word_freq	1	を	621
topic_word_freq	1	は	589
...
topic_word_sum	2	27892
topic_word_freq	2	の	1762
topic_word_freq	2	に	1378
topic_word_freq	2	を	1138
topic_word_freq	2	た	995
...
@data       # 訓練データと各文書に割り当てられたトピック(state)
# state 4 # comment
今日 は いい 天気 ...
```
