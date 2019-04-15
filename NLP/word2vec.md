# Word2vec
- Deep learning 기법
- word embedding to vector

###### 1. 소개
    1. 컴퓨터는 숫자만 인식할 수 있으므로, 한글/이미지 등을 바이너리 코드로 저장해야 한다.
    2. bag of word라는 개념을 이용하여 문자를 벡터화하여 머신러닝 알고리즘(딥러닝 알고리즘)이 이해할 수 있도록 벡터화하는 작업을 한다.
    3. one hot encoding or bag of word에서 vector size가 매우 크고 sparse하므로 neural network 성능이 잘 나오지 않는다.
    4. 이 때 사용한다.

###### 2. 특징
    1. 주위 단어가 비슷하면 해당 단어의 의미는 유사하다 라는 생각으로 접근
    2. 단어를 트레이닝시킬 때 주위 단어를 label로 매치하여 최적화
    3. 단어를 의미를 내포한 dense vector로 매칭시키는 것
    4. word2vec은 분산된 텍스트 표현을 사용하여 개념 간 유사성을 본다. 예를 들어, 파리와 프랑스가 베를린과 독일이(수도와 나라) 같은 방식으로 관련되어 있음을 의미한다.

###### 3. CBOW와 skip-gram
    1. 2가지 기법이 있다.
    2. CBOW(continuous bag-of-words)는 전체 텍스트로 하나의 단어를 예측하기 때문에 작은 데이터 세트일수록 유리하다.
    3. skip-gram의 경우 타겟 단어로부터 원본 단어를 역으로 예측하는 것이다. CBOW와는 반대로 컨텍스트-타겟 쌍을 새로운 발견으로 처리하고 큰 규모의 데이터셋을 가질 때 유리하다.
    


[관련코드](https://programmers.co.kr/learn/courses/21/lessons/1697)


###### 4. parameter
    1. Gensimd 패키지 사용
    2. [gensim: models.word2vec – Deep learning with word2vec](https://radimrehurek.com/gensim/models/word2vec.html)
    3. 아키텍처 : 아키텍처 옵션은 skip-gram(default) or cbow model 을 선택한다. skip-gram이 느리지만 더 좋은 결과를 내는 경향이 있다.
    4. 학습 알고리즘 : hierachical softmax(default) of negative sampling
    5. 빈번하게 등장하는 단어에 대한 down sampling : google document는 0.00001 ~ 0.001의 값을 권장
    6. 단어 벡터 차원 : 많은 features를 사용한다고 무조건 좋은 것은 아니다(overfitting). 합리적인 값은 수십에서 수백개 정도..
    7. 컨텍스트 / 창 크기 : 학습 알고리즘이 고려해야 하는 컨텍스트의 단어 수는 얼마나 될까? hierachical softmax를 위해 좀 큰 수도 좋지만, 10~수십개 정도..
    8. worker threads : 실행할 병렬 프로세스의 수, 컴퓨터마다 다르지만, 대부분 4~6의 값을 사용한다.
    9. 최소 단어 수 : 어휘의 크기를 의미 있는 단어로 제한하는 데 도움이 된다. 모든 문서에서 여러번 발생하지 않은 단어는 무시가 된다. 10~100이 적당하며, 높은 값은 제한 된 실행시간에 도움이 된다.
        
###### 5. 벡터 양자화(vector quantization)
    1. 벡터를 그룹화하는 것이다.
    ex) k-means 로 군집화를 하여 클러스터라는 단어의 중심을 찾는다.
    2. 이런 것이 가능한 이유는 Word2vec은 의미가 관련 있는 단어들의 클러스터를 생성하기 때문에, 클러스터 내의 단어 유사성을 이용하려는 것이다.
    3. k-means로 군집화하고, 랜덤포레스트를 사용하여 리뷰가 추천인지, 비추천인지 예측해 볼 수 있다.
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
