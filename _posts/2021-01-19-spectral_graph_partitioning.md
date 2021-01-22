---
title: "Spectral graph partitioning"
date: 2021-01-19
excerpt: "Laplacian matrix가 graph partitioning에 어떻게 활용되는지 알아보자"
excerpt_separator: "<!--more-->"
layout: single
classes: wide
read_time: false
tags: 
 - graph
---





## 다루는 내용
- spectral graph theory란 무엇인지
- laplacian graph의 정의와 성질은 어떠한지
- 위 내용들이 graph partitioning에 어떻게 활용되는지



## Spectral graph partitioning
#### Spectral graph theory

위키피디아에 spectral graph theory는 다음과 같이 소개되고 있다:  
In mathematics, spectral graph theory is **the study of the properties of a graph in relationship to the characteristic polynomial, eigenvalues, and eigenvectors of matrices associated with the graph,** such as its adjacency matrix or Laplacian matrix.

쉽게말하면 graph의 matrix representing(Adjacency matrix, Laplacian matrix 등..)의 spectrum을 분석하는 것이다.  
matrix에서 spectrum은 eigenvalue의 집합이므로 이 이론은 그래프를 나타내는 matrix의 eigenvalue, eigenvector를 이용해 graph를 조망한다.   

다양하게 응용되고 있는데, 오늘 다룰 spectral graph partitioning은 여러 응용 중 하나이다.


#### Laplacian matrix

$$L = D - A$$

위 식에서 보듯 Laplacian matrix 자체는 degree matrix와 adjacency matrix의 차에 불과하다.  
중요한 성질은 $\mathbf{x}^T L \mathbf{x}$를 재표현한 형태에서 나온다.

임의의 벡터 $\mathbf{x}$에 대하여,

$$\mathbf{x}^T L \mathbf{x} = \sum\sum L_{ij}\mathbf{x}_i\mathbf{x}_j $$

$$= \sum\sum(D_{ij}-A_{ij})\mathbf{x}_i\mathbf{x}_j$$

$$= \sum D_{ii} \mathbf{x}_i^2 - \sum_{(i,j) \in E} 2\mathbf{x}_i \mathbf{x}_j$$  

그리고 $$\sum D_{ii} \mathbf{x}_{i}^{2} = \sum_{(i,j) \in E} (\mathbf{x}_i^2+\mathbf{x}_j^2)$$ 이기 때문에,
{: style="text-align: center;"}

$$= \sum_{(i,j) \in E} (\mathbf{x}_i^2+\mathbf{x}_j^2 - 2\mathbf{x}_i \mathbf{x}_j) $$

$$= \sum_{(i,j) \in E}(\mathbf{x}_i - \mathbf{x}_j)^2 $$



위 식을 eigenvector $v$에 대해서 풀면, $v^TLv = \lambda v_Tv \geq 0$이기 때문에 Laplacian matrix는 positive semi-definite하다.  

#### Connectivity & eigendicomposition of Laplacian matrix 

Laplacian matrix의 eigenvalue와 eigenvector는 그래프의 connectivity와 깊은 상관이 있다.  
우선 가장 작은 eigenvalue, $\lambda_1$은 0의 값을 가지는데, 이는 L이 D-A라는걸 생각할 때 당연하다.  
그럼 $\lambda_1$에 대응하는 eigenvector는 어떨까?

아주 간단한 Laplacian matrix의 예시를 보자. 



$$ L_1 = 
    \begin{bmatrix}
    1 & -1 & 0 & 0 \\
    -1 & 1 & 0 & 0 \\
    0 & 0 & 1 & -1 \\
    0 & 0 & -1 & 0 \\
    \end{bmatrix}
$$

$$ L_2 = 
    \begin{bmatrix}
    3 & -1 & -1 & -1 \\
    -1 & 3 & -1 & -1 \\
    -1 & -1 & 3 & -1 \\
    -1 & -1 & -1 & 3 \\
    \end{bmatrix}
$$


$L_1$이 나타내는 것은 두 vertex씩만 연결되어 있는 disconnected graph이고, $L_2$의 경우는 connected graph이다.  

$L_1$의 $\lambda = 0 $에 대응하는 eigenvector를 생각해보면, 직교하는 두 벡터 $(0, 0, 1, 1)$과 $(1, 1, 0, 0)$가 있다. 
때문에 적어도 $\lambda_2$까지는 0의 값을 가진다. 또한, 0의 값을 가지는 $\lambda$의 multiplicity가 2임을 기억하자.  

한편, $L_2$의 경우  $\mathbf{x}^T L \mathbf{x} = \sum_{(i,j) \in E}(\mathbf{x}_i - \mathbf{x}_j)^2 = 0$ 을 이용하면, $v_1$은 모든 entry의 값이 동일한 벡터라는 걸 알 수 있다.  
이 경우 0의 값을 가지는 $\lambda$의 multiplicity는 1이다.


이 예를 통해 두 가지 사실을 확인했다.
1. $\lambda_2 > 0$, iff G is connected
2. 가장 작은 eigenvalue의 multiplicity는 그래프의 connected components 수와 같다.


#### Graph partitioning

다시 graph partitioning 문제로 돌아와보자.  
graph partitioning 문제는 각 vertex에 1 혹은 -1 값을 부여하는 문제로 전환되어, 아래와 같은 식으로 나타낼 수 있다.  

$$\min_x \sum_{(i,j)\in E} (\mathbf{x}_i - \mathbf{x}_j)^2$$

이 때, $x_i$는 assignment vector로 다음과 같이 정의된다.  

 $$  \mathbf{x}_i =
\begin{cases}
\begin{aligned}
 1,  & \text{ node $i \in S$} \\
-1, & \text{ node $i \in \bar{S}$}
\end{aligned}
\end{cases}$$  

각 vertex에 해당하는 $x_i$에 모두 같은 값을 부여하면 위 식이 0으로 최소화되겠지만, partition을 찾으려는 우리의 목적에 어긋나기 때문에 다음과 같은 constraint을 건다.

$$\sum_i \mathbf{x}_i = 0 $$  

잎에서 보았듯, $ \sum_{(i,j)\in E} (\mathbf{x}_i - \mathbf{x}_j)^2 = \mathbf{x}^TL\mathbf{x}$ 이므로, 재표현하면 다음과 같다.

$$
\begin{array}{ll}
\text{minimize}  & \mathbf{x}^TL\mathbf{x}  \\
\text{subject to}& \mathbf{x}^T\mathbf1=0 \\
& \mathbf{x}_i \in \{ -1, 1 \}
\end{array}
$$  


이 문제는 NP-hard이므로, $\mathbf{x}_i \in \\{ -1, 1 \\}$의 constraint을 $\mathbf{x}^T\mathbf{x}=n$으로 느슨하게 해준다.  

relaxed constraint 하에서 최적화 문제를 풀면, vector $\mathbf{x}^*$는 Laplacian matrix의 두 번째로 작은 eigenvalue에 대응하는 eigenvector이다. 이 벡터는 Fiedler vector라고도 불린다.  

이렇게 partitioning의 문제가 Laplacian matrix의 eigenvalue를 찾는 문제로 전환되었다.




#### Reference

[Lecture note: CS 224W - Graph Clustering by Austin Benson](http://snap.stanford.edu/class/cs224w-2016/slides/clustering.pdf)  
[Lecture note: Spectral Graph Theory - The Laplacian by Daniel A. Spielman](https://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf)  
[Mining Massive Datasets - Stanford University,  lecture 30 - 33](https://youtu.be/FRZvgNvALJ4)  
[What does the value of eigenvectors of a graph Laplacian matrix mean?](https://math.stackexchange.com/questions/3853424/what-does-the-value-of-eigenvectors-of-a-graph-laplacian-matrix-mean)  