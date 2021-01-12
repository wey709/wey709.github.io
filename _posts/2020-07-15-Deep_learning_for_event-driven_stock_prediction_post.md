---
title: "논문 공부: Deep learning for event-driven stock prediction"
date: 2020-07-15
excerpt: "20200715 세미나 주제"
excerpt_separator: "<!--more-->"
layout: single
classes: wide
tags: 
 - deep learning
 - embedding
 - reading papers 
---




공부한 논문: Ding, Xiao, et al. "Deep learning for event-driven stock prediction." Twenty-fourth international joint conference on artificial intelligence. 2015.

주가 예측을 할 때 관련 뉴스를 가지고 NLP를 사용한다는건 평범한 접근 방법인데, 그에 한 단계 나아가 주가 관련 이벤트를 추출하고, 임베딩했다는 점이 새로웠다.
이벤트 추출할 때 쓰는 기술들이 open information extraction이라던가 큰 주제에 종속되어 있어서 그 부분에 대해서는 가볍게 보고 넘어갔다.

## Event extraction & embeddings

#### Event extraction
이벤트를 튜플 \\(E\\)로 나타낸다. 
\\[E = (O_1, P, O_2, T)\\]
\\(P\\)는 action이며 \\(O_1, O_2\\)는 Object이다. 예를 들어 'Samsung display unit sees robust smartphone growth'라는 문장이 있다면 \\(E\\)는 (Samsung display unit, sees, robust smartphone growth)가 되는 것이다.
이벤트 추출을 위해서는 ReVerb와 ZPar를 사용하는데, 우선 ReVerb로 후보 튜플 \\(E'\\)를 뽑고, ZPar로 타당한지 필터링한다. 
이벤트 추출을 하면 모든 문장은 각각 다른 이벤트 \\(E\\)로 추출되어, 전체적으로 봤을 때 매우 sparse한 구조가 되고 이를 해결하기 위해 embedding을 하는 것이다. 


#### Event embedding
추출한 \\(E\\)의 \\(O_1\\), \\(O_2\\), \\(P\\)를 skip-gram을 이용해 100차원 벡터로 만들고, 그에 맞춰 (100,100,K) 텐서 \\(T_1\\), \\(T_2\\)를 생성한다.
\\[{R_1} = f(O^T_1T_1^{[1:k]}P+W    \begin{bmatrix}     O_1 \\    P \\ \end{bmatrix} +b)\\]이고, \\(R_2\\)는 \\(O_2\\),\\(T_2\\), \\(P\\)를 이용해 유사하게 계산한다.
최종 임베딩 벡터인 \\(U\\)는 \\(R_1\\)과 \\(T_3\\) 그리고 \\(R_2\\) 이용해 유사하게 계산한다.
\\(T_1\\)과 \\(T_2\\)가 \\(O_1\\)과 \\(P\\), \\(P\\)와 \\(O_2\\) 사이의 관계를, \\(T_3\\)가 \\(R_1\\), \\(R_2\\) 관계를 포착하도록 학습된다.
즉 \\(T_1\\), \\(T_2\\), \\(T_3\\), \\(W\\), \\(b\\)는 모두 학습되는 파라미터이다. 

학습을 할 때는 \\(E\\)에 대응되는 \\(E^r\\) 튜플을 만드는데 \\(P\\)와 \\(O_2\\)는 정상적인 튜플\\(E\\)의 \\(P\\)와 \\(O_2\\)와 같고, \\(O_1\\)을 무작위로 대체한다. 
\\(E\\)와 \\(E^r\\) 각각 내부의 원소들끼리의 관계가 얼마나 그럴듯한지 계산하는데(당연히 무작위로 만들어진 튜플 \\(E^r\\)이 덜 그럴듯할 것이다), 두 score의 차이가 1 미만이면 back-propagation을 통해 가중치(학습되어야하는 파라미터들)를 갱신하고, 1 이상이 되면 그 튜플에 대해서는 학습을 멈추고 다음 튜플로 넘어간다.

NTN model은 Socher, et al의 "Reasoning with neural tensor networks for knowledge base completion." 논문에 자세히 나온다. 
왜 텐서를 쓰는지는 위 논문에서 이야기하는데, 예를 들어 그냥 neural network에 개체(여기서는 \\(P\\)와 두 개의 \\(O\\))가 들어간다면 두 벡터는 (비선형적인 activation function을 통과한다 한들) concatenate 될 뿐인데, 중간에 텐서를 놓으면 그 텐서의 각 slice가 두 벡터의 관계를 포착할 수 있고, 여러 slice를 두기 때문에 다양한 관계를 표현가능하다는 것이다.
그래서 has part라는 말이 있을 때 car has part~ 와 dog has part~는 조금 상이한 의미를 가지는데 그 두 가지 의미를 각각 다른 slice로 표현시킬 수 있다고 논문에서 예시로 든다.


## Deep Prediction Model
임베딩한 이벤트로 주가를 예측하는데, 특이하게도 CNN을 쓴다. 30일 이내인 long-term events들끼리 일정한 크기의 윈도우로 convolution 연산을 해 long-term events들의 feature를 뽑은 벡터가 만들어지고, 7일 이내의 mid-term events들도 마찬가지 방식으로 벡터가 만들어진다. 주가를 예측하는 날 하루 전의 short-term events들은 각각 임베딩된 \\(U\\) 그대로 사용된다. 이 벡터들이 hidden layer를 통과해 output으로는 증가/감소의 binary class가 나온다. 


## Experiments
두 가지 측면에서 비교하는데, 하나는 event-embedding이 word-embedding이나 임베딩하기 전의 event extraction보다 나은지 비교하는 것이고, 다른 하나는 vanilla neural net을 쓰는 것보다 CNN을 쓰는게 나은지 비교하는 것이다. 또 다른 SOTA와도 비교한다. 결과적으로 같은 조건일 때 input으로 event-embedding이 제일 낫고, 같은 input일 때 NN보다 CNN이 낫다. 특히 눈여겨 볼 점은 individual stock prediction 부분인데 Fortune magazine의 랭킹이 낮은 회사일수록 다른 방법들과 이 논문의 방법(event embedding & cnn)의 accuracy가 baseline과 비교했을 때 크다는 것이다. 랭킹이 낮을수록 관련 뉴스도 적게 뜰 것인데 CNN 덕분에 비교적 넓은 기간의 이벤트들까지 고려할 수 있게되어서 그런 것이라고 해석한다.
