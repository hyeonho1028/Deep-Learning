### NLP

정의 : 직역하면 자연어 처리이며, 여러가지 과정을 거친다.

###### 1. DATA CLEANING & TEXT PROCESSING
    - ex) BeautifulSoup을 통해 html 태그 제거
    - 정규표현식으로 알파벳 이외의 문자를 공백으로 치환
    - ex) NLTK 데이터를 사용하여 불용어(stopword)를 제거
    - 어간추출(stemming), 음소표기법(lemmatizing)
    - stemming방법 중 하나인 Snowballstemmer를 통해 어간을 추출

###### 2. TEXT DATA PREPROCESSING
    - normalization(정규화)
        ex) 입니닼, 입니답, 입니다 -> 입니다., 샤랑해, 샤릉해 -> 사랑해
    - tokenization(토큰화)
        ex) 한국어를 자르는 느낌입니다.ㅋㅋ -> 한국어Noun, 를Josa, 자르Noun, 는Josa, 느낌Noun, 입Adjective, 니다Eomi, ㅋㅋKoreanParticle
    - stemming(어근화)
        ex) 한국어를 자르는 느낌입니다. ㅋㅋ -> 한국어Noun, 를Josa, 자르다Verb, 느낌Noun, 이다Adjective, ㅋㅋKoreanParticle
    - phrase extraction(어구 추출)
        ex) 한국어를 자르는 느낌입니다. ㅋㅋ -> 한국어, 자르, 느낌, 자르는 느낌
        
        
###### 3. 정규표현식
```
# 정규표현식을 사용해서 특수문자를 제거
import re
# 소문자와 대문자가 아닌 것은 공백으로 대체한다.
letters_only = re.sub('[^a-zA-Z]', ' ', example1.get_text())
```

###### 4. 토큰화
```
# 모두 소문자로 변환한다.
lower_case = letters_only.lower()
# 문자를 나눈다. => 토큰화
words = lower_case.split()
```

###### 5. STOPWORD(불용어) 처리
일반적으로 코퍼스에서 자주 나타나는 단어는 학습 모델로서 학습이나 예측 프로세스에 실제로 기여하지 않아 다른 텍스트와 구별하지 못한다. 예를 들어 조사, 접미사, i, me, my, it, this, that, is, are 등과 같은 단어는 빈번하게 등장하지만, 실제 의미를 찾는데는 크게 기여하지 않는다. 그러므로 제거 하는데 stopwords는 'to' or 'the'와 같은 용어를 포함하므로 사전 처리 단계에서 제거하는 것이 좋다. NLTK에는 153개의 불용어가 미리 정의되어 있다. 17개의 언어에 대해 정의되어 있으며 한국어는 없다고 한다.(+ 한국어 불용어가 있는 사전을 찾아볼 필요성이 있다.)

```
import nltk
from nltk.corpus import stopwords
stopwords.words('english')[:10]
```
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']

```
# stopwords 를 제거한 토큰들
words = [w for w in words if not w in stopwords.words('english')]
print(len(words))
words[:10]
```
이런 느낌

###### 6. stemming(스테밍, 어간 추출, 형태소 분석)
출처 : 어간 추출 - 위키백과, 우리 모두의 백과사전

어간 추출(語幹 抽出, 영어: stemming)은 어형이 변형된 단어로부터 접사 등을 제거하고 그 단어의 어간을 분리해 내는 것
message, messages, messaging과 같이 복수형, 진행형 등의 문자를 같은 의미의 단어로 다룰 수 있도록 도와준다.
stemming(형태소 분석): 여기에서는 NLTK에서 제공하는 형태소 분석기를 사용한다. 포터 형태소 분석기는 보수적이고 랭커스터 형태소 분석기는 좀 더 적극적이다. 형태소 분석 규칙의 적극성 때문에 랭커스터 형태소 분석기는 더 많은 동음이의어 형태소를 생산한다.
```
# 포터 스태머의 사용 예
stemmer = nltk.stem.PorterStemmer()
print(stemmer.stem('maximum'))
print("The stemmed form of running is: {}".format(stemmer.stem("running")))
print("The stemmed form of runs is: {}".format(stemmer.stem("runs")))
print("The stemmed form of run is: {}".format(stemmer.stem("run")))
```
maximum
The stemmed form of running is: run
The stemmed form of runs is: run
The stemmed form of run is: run

```
# 랭커스터 스태머의 사용 예
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
print(lancaster_stemmer.stem('maximum'))
print("The stemmed form of running is: {}".format(lancaster_stemmer.stem("running")))
print("The stemmed form of runs is: {}".format(lancaster_stemmer.stem("runs")))
print("The stemmed form of run is: {}".format(lancaster_stemmer.stem("run")))
```
maxim
The stemmed form of running is: run
The stemmed form of runs is: run
The stemmed form of run is: run

###### 7. Lemmatization(음소표기법)
언어학에서 음소 표기법 (또는 lemmatization)은 단어의 보조 정리 또는 사전 형식에 의해 식별되는 단일 항목으로 분석될 수 있도록 굴절된 형태의 단어를 그룹화하는 과정이다.
예를 들어 동음이의어가 문맥에 따라 다른 의미가 있는데

    1) 배가 맛있다.
    2) 배를 타는 것이 재미있다.
    3) 평소보다 두 배로 많이 먹어서 배가 아프다.

위에 있는 3개의 문장에 있는 배는 모두 다른 의미가 있다. 

레마타이제이션은 이때 앞뒤 문맥을 보고 단어의 의미를 식별하는 것이다.
영어에서 meet는 meeting으로 쓰였을 때 회의를 뜻하지만, meet일 때는 만나다는 뜻을 갖는데 그 단어가 명사로 쓰였는지 동사로 쓰였는지에 따라 적합한 의미가 있도록 추출하는 것이다.
```
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

print(wordnet_lemmatizer.lemmatize('fly'))
print(wordnet_lemmatizer.lemmatize('flies'))

words = [wordnet_lemmatizer.lemmatize(w) for w in words]
# 처리 후 단어
words[:10]
```

###### 8. 문자열 처리
```
def review_to_words( raw_review ):
    # 1. HTML 제거
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. 영문자가 아닌 문자는 공백으로 변환
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. 소문자 변환
    words = letters_only.lower().split()
    # 4. 파이썬에서는 리스트보다 세트로 찾는 게 훨씬 빠르다.
    # stopwords 를 세트로 변환한다.
    stops = set(stopwords.words('english'))
    # 5. Stopwords 불용어 제거
    meaningful_words = [w for w in words if not w in stops]
    # 6. 어간추출
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 7. 공백으로 구분된 문자열로 결합하여 결과를 반환
    return( ' '.join(stemming_words) )
```
이런 느낌으로 처리
    
 
 
 
[Cambridge University Press](https://nlp.stanford.edu/IR-book/html/htmledition/contents-1.html)
