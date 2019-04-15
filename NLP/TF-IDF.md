# TF-IDF
- TF-IDF를 이용하여 벡터화를 해보자.

###### 1. 정의
    1. TF-IDF란 TF(Term Frequency)로서 단어빈도를 의미한다. 특정한 단어가 문서 내에 얼마나 자주 등장하는지를 나타낸 값으로, 이 값이 높을 수록 문서에서 중요하다고 생각할 수 있다.
    2. 하지만 단어 자체가 문서군 내에서 자주 사용되는 경우, 이것은 그 단어가 흔하게 등장한다는 것을 의미하고, 이것을 DF라고한다.(document frequency)
    3. DF 이 값의 역수를 IDF라고 한다. 즉 TF-IDF는 TF * IDF를 의미한다.
    4. IDF 값은 문서의 성격에 따라 결정되는 경향이 있다. 예를 들어 '원자'라는 낱말은 일반적인 문서들 사이에서는 잘 나오지 않기 때문에 IDF값이 높아지고 문서의 핵심어가 될 수 있지만, 원자에 대한 문서를 모아놓은 문서군의 경우 이 낱말은 상투어가 되어 각 문자들을 세분화하여 구분할 수 있는 다른 낱말들이 가중치를 얻게 된다.
    5. 역문서 빈도(IDF)는 한 단어가 문서 집합 전체에서 얼마나 공통적으로 나타내는지를 나타내는 값이다. 전체 문서의 수를 해당 단어를 포함한 문서의 수로 나눈 뒤 로그를 취하여 얻을 수 있다.

출처 : [위키백과TF-IDF](https://ko.wikipedia.org/wiki/Tf-idf)

###### 2. python function
    1. TfidfTransformer()
    2. norm='l2' 각 문서의 피처 벡터를 어떻게 벡터 정규화 할지 정한다.
    3. L2 : 벡터의 각 원소의 제곱의 합이 1이 되도록 만드는 것이고 기본 값
    4. L1 : 벡터의 각 원소의 절댓값의 합이 1이 되도록 크기를 조절
    5. smooth_idf=False : 피처를 만들 때 0으로 나오는 항목에 대해 작은 값을 더해서(스무딩을 해서) 피처를 만들지 아니면 그냥 생성할지를 결정
    6. sublinear_tf=False : TF-IDF를 사용해 피처를 만들 것인지 아니면 단어 빈도 자체를 사용할 것인지 여부
    7. use_idf=True : TF-IDF를 사용해 피처를 만들 것인지 아니면 단어 빈도 자체를 사용할 것인지 여부


[관련코드](https://programmers.co.kr/learn/courses/21/lessons/1846)