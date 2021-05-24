---
title: "BERT(feature-based)로 sequence tagging할 때 주의점"
date: 2021-05-24
excerpt: "다른 말로 삽질기록"
excerpt_separator: "<!--more-->"
layout: single
classes: wide
read_time: false
tags: 
 - 삽질
---





### Overall
어차피 베이스라인으로 쓰려면 기본적으로 BERT를 쓰는 모델로 이것저것 해봐야하기도 하고, 익숙하지 않은 파이토치 연습도 할 겸해서 BERT 논문 5.3절에 있는 Feature-based apporach를 시도해보았다. 그 중에서도 마지막 은닉층만 쓰는걸로 했는데 뭐 하루정도 하면 끝나겠지 했던걸 생각보다 많은 시간을 소요한 끝에 논문과 엇비슷한 score가 나왔다... 하면서 얼마나 내가 대충 알고있는지 깨달았기에 몇 가지 알게된 점을 기록해본다.


### Lessons learned


1. CLS, SEP 토큰   
sequence tagging 태스크들에는 CLS, SEP 토큰이 필요하지 않다. 근데 BERT 모델 자체가 저것들을 포함해 학습되었으므로 저 토큰들을 포함해서 학습하는게 무조건 낫다. 또 중요한건 CLS, SEP, PAD 각각 다른 label을 줘야한다는 점이다. 어래에서 더 자세히 언급한다.
2. Loss function   
이걸 하려고 결심했을 때 제일 처음 고민한 부분이다. loss function은 결국 모델이 뭘 학습해야할지를 반영해야하는데, 패딩된 부분들을 맞추는 것은 학습의 범위가 아니다. 따라서 true label이 PAD인 것들을 마스킹해, loss function에 반영되지 않게 해줘야 한다. **그런데 CLS와 SEP은 다르다..** 내가 저지른 실수도 이거였다. CLS, SEP, PAD을 모두 같은 취급했고 마스킹했다. BERT가 어떻게 학습되는지 생각해볼 때 그러면 안되고, 각각 다른 label을 부여해야하며 저 둘은 loss function에 반영되도록 해야한다.
3. 알맞은 BERT 모델   
나는 NER을 했기 때문에 BERT base 중에서도 cased를 써야 했는데 (관성적으로) uncased를 썼었다. 고유명사 중에 대문자로 시작되는 것들이 얼마나 많은지 생각해보면 당연히 cased로 해야 낫다.
4. Randomness를 제어하자   
이건 성능 때문인건 아니고 실험이라면 당연히 통제해야하는 부분이다. 구글링 해보면 파이토치에서 모든 랜덤을 제어하는 아주 좋은 포스팅이 있다.


처음에 고전하다가 3을 고치니까 20 정도 오르고 2까지 고치니까 f1 score가 10정도 올랐었다. 그것 때문인지 모르고 learning rate 탓인줄 알고 엄청난 세월을 낭비했는데 진짜 택도 아닌 성능이 나오면 구조 먼저 의심해보는게 합리적이다.


