My Idea
- textureによるクラスタリング
- DA + Dropoutによる不確かさ判定 (uncertainty + version space探索）
- KFAC??(きびしいかもしれない)
- active + semi-super
    - 不確かなやつにlabel, 確かなやつにpseudo label
- 位置によるごちゃごちゃ
- 最終層のヘッシアンのみ利用する


Experiment
- Mnist実験まとめ
    - 6400sample毎 (25, 64, 100, 400 iteration毎) に結果出力
        - 収束速度は大体同じ
        - 最終的なtest scoreは25毎 (つまりiteration回数が小さくbatchsizeを大きくしたほうがよかった
    - 見るsample数固定してiteration回数変えてみたら、batchsize大きい方が良かった
- Batch size関係なく割とiteration回数に依存しそう
 

1000 : 0.55830
5000 : 0.78150
10000 : 0.85140
30000 : 0.90810
50000 : 0.92100


Survey


icml 2009資料
http://hunch.net/~active_learning/active_learning_icml09.pdf


2009 settle’s survey
http://apophenia.wdfiles.com/local--files/start/settles_active_learning.pdf


日本語訳
http://blog.livedoor.jp/itukano/archives/51826566.html
http://blog.livedoor.jp/itukano/archives/51826569.html


theory of active learning
https://docs.google.com/viewer?url=http://www.stevehanneke.com/docs/active-survey/active-survey.pdf


誰かの講演
https://www.cs.huji.ac.il/~shais/Lecture1.pdf banditとの関連
https://www.cs.huji.ac.il/~shais/Lecture2.pdf QBCの理論
https://www.cs.huji.ac.il/~shais/Lecture3.pdf disagreement coefficient & agnostic active


agnostic active
http://www.cs.cmu.edu/~ninamf/papers/agnostic-active-icml.pdf


importance weighted active learning
http://cseweb.ucsd.edu/~dasgupta/papers/iwal-icml.pdf


hierarchical sampling for active learning
http://www.cs.columbia.edu/~djhsu/papers/hier.pdf


Theory of Disagreement-Based Active Learning
http://www.nowpublishers.com/article/Details/MAL-037




- sampling bias figure このとき絶対左から二番目のサンプルが選ばれない





応用先
- cancer classification
- malware detection
-  autonomous driving [3, 4]


# Active Learning教科書


- 問題設定
    - stream-based 次々にデータが来て、なんらかのutilityに対して閾値を超えればlabelling
    - pool-based Unlabelled dataがあってそこから一番utilityの高いもの
    - query synthesis いい感じの人工データを作る


- Uncertainly Sampling
    - 一応やりたいこととしてはversion spaceの効率的な探索とも言える (version spaceがsymmetricなら）
    - 不確かなデータを選ぶ
        - least confident: 確率最大ラベルの確率が小さいやつ
        - margin sampling: 確率最大ラベルの確率と、次に大きいラベルの確率の差が一番小さいやつ
        - entropy-based: エントロピー最大
    - sampling biasが問題


- Efficient search through the hypothesis space
    - version spaceを最も小さくするサンプル
    - Query-By-Disagreement: version spaceを保持
    - Query-By-Committee : 複数のclassifierがあって多数決で票が割れたやつ (アンサンブル前提）
        - できるだけ性格の違うやつがいい(適当にsample作って片方はpositive labelにしてpositiveが出やすく
        - Vote Entropy: 投票結果のエントロピーが大きい
        - average KL: 学習器の予測分布と、それらの平均したものとのKLの平均が大きい
           - 各学習器の出力がばらばらだと大きくなる


- Exploiting Data structure
    - Density-Weighted Methods 分布を考えて、データが密なところから取ってくる
        - Information Density: 類似度が大きいデータが多い順 (類似度が高いデータ上位数件の平均距離)
    - hierarchical sampling
        - いいかんじの階層クラスタリングしてpseudo labelling


- Expected Model Change, Expected Error Reduction.
    - 新しいラベルがyだったとしたときの期待値を取ってモデルがどう変わるかをゴリゴリ計算
    - re-trainに時間がかかると厳しい


-  Variance Reduction
    - 新しいラベルによってモデル内分散がどう変わるかをゴリゴリ計算
    - 汎化誤差 = ノイズ + バイアス + モデル内分散
    - Cramel-rao inequality
        - フィッシャー情報とモデル内分散には相関がある
        - フィッシャー情報の逆数を小さくすることがモデル内分散を小さくすることに


- 行列を小さくする手法3種類
■A-optimality
行列の trace を最小に <-フィッシャー情報ではこれ
■D-optimality
行列の determinant を最小に (Query-By-Committeeで使われる？)
■E-optimality
行列の最大固有値を最小に


- 結局A-optimalityが使われるらしい
    - unlabellのFIMとlabelled+xのFIMの逆行列のdot product (fisher information ratio)が小さいやつをとってくる
- ちょっと理解できなかったが、Submodularity？的な単調性からbatch active learningに自然に拡張されるらしい
- 途中出てきたmatrix inversion identities (http://www0.cs.ucl.ac.uk/staff/g.ridgway/mil/mil.pdf)




# Adaptive Submodular function


- Adaptive Submodularity: Theory and Applications in Active Learning and Stochastic Optimisation
- Near-optimal batch mode active learning and adaptive sub modular optimisation
    - たぶんこの二つがactive learningへの劣モジュラ適用のpioneer


- Adaptive submodular optimization under matroid constraints
    - matroidに制限を加えた時にも劣モジュラを利用できる


- Interactive Submodular Set Cover 
- Near-Optimal Bayesian Active Learning with Noisy Observations 
- Agnostic Active Learning Without Constraints
- Statistical Active Learning Algorithms
最近のいい感じの論文って感じで載ってたがよくわからんのが多い


# Scalable Histopathological Image Analysis via Active Learning http://www.ee.columbia.edu/~wliu/MICCAI14_active.pdf
- 適応的劣モジュラ関数を設計 (適応の要素の意味よくわからん、目的関数が変化するだけかな）
    - 現在のversion spaceからsamplingした (hit-and-run sampling) Hがxによって減る期待値
    - 上記の期待値をscoreとして最大となるものをgreedyに選ぶ
    - その際pathologyのheuristicsで、似てるやつ多いはずだから別のpartitionから取ってくる
    (partitionはk-meansで)
        - partition matroid constraintとよんでいる
    - 10000次元のtexture特徴量から線形識別器
    - scalable要素はない


# Deep Bayesian Active Learning with Image Data
https://arxiv.org/pdf/1703.02910.pdf
(MNIST + ISIC2016(皮膚がん))


## 背景
- 基本的にはラベル付けは少ない事が多い。特徴量抽出にはもっといっぱい必要
- many AL acquisition functions rely on model uncertainty.  
But in deep learning we rarely represent such model uncertainty
- pretrain-modelにもとに戻して再学習


- BALD Bayesian active learning for classification ´ and preference learning (関連論文)


- aleatoric uncertainty
事象の偶発性、ノイズに起因するもの
- epistemic
知識の欠如による認識論的な不確かさ




# CEAL
https://arxiv.org/pdf/1701.03551.pdf
(CACD + Caltech256)
## 背景
CNNはfeature learning+classifier training
- 基本的にはラベル付けは少ない事が多い。特徴量抽出にはもっといっぱい必要
- assumption that features is fixed


- uncertainty-based
- information density
    - average distance with other samples within a cluster 
- diversity
    - calculated by clustering the most uncertain samples via k-means with histogram intersection kernel.


## 提案手法
- informative minority + majority high confidence
- more powerful + more discriminative
- high confidenceの定義はentropyが小さいこと。そのthresholdは徐々に小さくしていく。




# Single-pass active learning with conflict and ignorance.
 https://link.springer.com/content/pdf/10.1007%2Fs12530-012-9060-7.pdf


# Unbiased online active learning in data streams.
- streamで来るデータをどうするか
- 今まで見たことあるやつと似てるかどうか、不確かかどうか程度




ICML 2017 
# Active Learning for Cost-Sensitive Classification
https://arxiv.org/pdf/1703.01014.pdf
- multi-class cost sensitiveな問題設定
- よくわからんけど、disagreement-based


# Diameter-Based Active Learning
https://arxiv.org/pdf/1702.08553.pdf
- splitting index?
- version space系
- ある仮説分布hからsamplingした仮説hをxのpredictionによってV+, V-に分類し、V+内の分散、V-内の分散が小さければそのサンプルによって分散が小さくなるというIdea
- non-para active
    - An efficient graph based active learning algorithm with application to nonparametric classification


# Active Learning for Accurate Estimation of Linear Models
- UCBとか使ってるからlinearの理論系（たぶん）


# Learning Algorithms for Active Learning
(Omniglot)
- active learningの方法を学習するmeta-learning 
- 関連# Active One-shot Learning https://arxiv.org/pdf/1702.06559.pdf
    - RNN-basedのaction-value function導入してラベルを要求するタイミングを強化学習で制御
    - One-shot learning with memory-augmented neural networks の論文と割りと近いらしい
    - modelがrequestするまでtrue labelを保留するactiveにしたのが新しいところ
    - active + RLはうちが初


# Active Learning for Top-KK Rank Aggregation from Noisy Comparisons
# Just Sort It! A Simple and Effective Approach to Active Preference Learning
- active ranking algorithm




CVPR 2017
# Non-Uniform Subset Selection for Active Learning in Structured Data
(SUN Dataset (scene classification), CORA dataset (document classification) VIRAT (activity))
- belief propagation使ってる
- relationship between data samplesを考慮 (これがわかったらこれもわかる、じゃあこれにラベル付)
- ちょっとむずそう


- 発想近い (exploiting data structure)
    - M. Hasan and A. K. Roy-Chowdhury. Context aware active learning of activity recognition models. ICCV IEEE, 2015
    - Link-based active learning. In NIPS Workshop on Analyzing Networks and Learning with Graphs, 2009


# Fine-tuning Convolutional Neural Networks for Biomedical Image Analysis: Actively and Incrementally∗
- finetune -> active sample -> finetuneの仕組みを謎に新規性として推してる
- data augmentationした複数の画像のprediction結果を利用
    - めっちゃ分散があったら使える
    - 全部uncertainだったら使える
    - 両極端だったらlabel noiseを生み出しがちだから使えないかも


- 上記の決定方法だからinitial samplesいらない
- また、何割かしか使わないことで計算量削減


# INTRODUCING ACTIVE LEARNING FOR CNN UNDER THE LIGHT OF VARIATIONAL INFERENCE
https://openreview.net/pdf?id=ryaFG5ige
(MNIST + USPS)
- 割とやりたいこと近くて悲しい論文
- 式変形が若干わからない
- KFACでhessianのdiagonal 近似かつkronecker近似 


## Large-scale Active Learning with Approximations of Expected Model Output Changes
(MS COCO)
- EMOCのいい感じのやつ (あんま読んでない）


Deep × Active


- Active Deep Networks for Semi-Supervised Sentiment Classification
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.690.5790&rep=rep1&type=pdf
(めっちゃ簡単)


- A new active labeling method for deep learning
(MNIST)
-RBMでpre-training
least confidenceとか超簡単死ぬほど簡単
http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6889457


- Active learning for hyperspectral image classification with a stacked autoencoders based neural network
(よくわからんdataset)
Auto encoderでpretraining
http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8075429


- Deep learning approach for active classification of electrocardiogram signals
- Captcha recognition with active deep learning.
https://pdfs.semanticscholar.org/ab6d/a3000aba197571d69b3e205a7945dd691469.pdf?_ga=2.156186555.1070462067.1510657515-688530405.1510657515


# Related work (その他）


## general
- A novel active learning method in relevance feedback for content-based remote sensing image
- Active learning for ranking through expected loss optimization,” IEEE Trans. 2015.
- A convex optimization framework for active learning,”  ICCV, 2013


## Image categorisation
- Multi-level adaptive active learning for scene classification,” in ECCV, 2014
- Multilabel image classification via high-order label correlation driven active learning,” IEEE Trans. Image Processing, vol. 23, 2014


- Active learning for semantic segmentation with expected change


- Context aware active learning of activity recognition models. IEEE, ICCV 2015


- X. Li and Y. Guo, “Adaptive active learning for image classification,” in CVPR, 2013
GP with RBF (uncertainty確保)


- Entropy-Based Active Learning for Object Recognition cvpr 
http://www.vision.caltech.edu/publications/holub_et_al_active_learning_cvpr_workshop.pdf
カーネル法？？


- Robust multi-label image classification with semi-supervised learning and active learning,” in MultiMedia Modeling, 2015.


- Scalable active learning for multiclass image classification,” IEEE Trans.
http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6127880


- Multi-Class Active Learning for Image Classification Ajay J. Joshi∗ ©2009
http://projectsweb.cs.washington.edu/research/insects/CVPR2009/optim_learning/multicls_activelerarning_classif.pdf
(Caltech 101 + UCI datasetから適当に)
カーネル法？グラフ？


## self-paced learning
- Self-paced curriculum learning,” in AAAI, 2015. 
- Self-paced learning for matrix factorization,” in AAAI, 2015.
- Easy samples first: Self-paced reranking for zero-example multimedia search,” in ACM Multimedia, 2014. 
- Self-paced learning with diversity,” in NIPS, 2014.


## curriculum learning
- Self-paced learning for latent variable models,” in NIPS, 2010. 
- Curriculum learning,” in ICML, 2009.


## bayesian 
- Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks
https://arxiv.org/pdf/1502.05336.pdf


## activeにする正当性
- Are all training examples equally valuable? 







