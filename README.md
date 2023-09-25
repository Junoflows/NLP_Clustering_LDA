# NPL_Clustering_LDA
2023-09-22부터 25일까지 데이콘에서 실시한 뉴스 기사 레이블 복구 해커톤 참여한 코드 및 정리 기록입니다.

## 대회 개요 및 설명
### 배경
데이콘 데이터 사이언티스트 chanmuzi가 아주 중요한 데이터셋의 레이블을 전부 날려버렸어요!  
4일 후에는 이 데이터셋을 활용한 프로젝트가 예정되어 있어 시간이 정말 촉박하고,  
그렇다고 4일동안 이 많은 샘플들을 직접 확인하여 손으로 모두 Labeling을 하기에는 현실적으로 불가능합니다.    
여러분의 통계적 지식과 인공지능 지식을 발휘하여 chanmuzi를 이 위기에서 구해주세요!   

### 설명
4일 내에 사용 예정인 이 데이터셋은 6개의 카테고리로 분류되어야 하는 6만 행의 csv 파일입니다.  
이 파일에는 'id'와 'text' 필드만 있을 뿐, 카테고리 정보는 사라져 버렸습니다.   
여러분의 목표는 이 'text'가 어떤 카테고리에 속하는지를 최대한 정확하게 예측하는 것입니다!  
(제공되는 데이터셋은 단 하나의 csv파일이며 카테고리 정보가 포함되지 않습니다)  
아래 표를 참고하여 데이터셋의 'category' 필드를 복구해주세요!  

![image](https://github.com/Junoflows/NPL_Clustering_LDA/assets/108385417/297255df-e0fd-4118-8423-c322807a30a3)

### 데이터
news.csv [파일]  
id : 샘플 고유 id  
title : 뉴스 기사 제목  
content : 뉴스 기사 전문

## 데이터 분석 과정

### 텍스트 전처리
+ 텍스트 처리
```
def clean_text(text):
    
    # URL 링크 뒤에 short_description 이 붙은 기사가 많음 -- 뉴스 기사에 큰 영향을 미치지 않으므로 제거
    text = text.replace('short_description','')
    
    # URL 제거
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 해시태그 제거
    text = re.sub(r'#\w+', '', text)
    # 멘션 제거
    text = re.sub(r'@\w+', '', text)
    # 이모지 제거
    text = text.encode('ascii', 'ignore').decode('ascii')
    # 공백 및 특수문자 제거
    text = re.sub(r'[^\w\s]', '', text).strip()
    # 숫자 제거
    text = re.sub(r'\d+', '', text)
    # 같은 글자가 3번 이상 연속으로 나타나는 패턴을 찾아서 1번으로 대체
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    # 소문자 변환
    text = text.lower()

    # 단어 원형으로 변환 후 시행
    # 불용어 제거 및 2글자 미만 단어 제거
    # text = ' '.join(word for word in text.split() if word not in stop_words and len(word) > 2)

    return text
```

+ 단어 원형으로 변환 Lemmatization
```
from nltk.stem import WordNetLemmatizer
import nltk
import string

# 단어 원형 추출 함수
lemmar = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmar.lemmatize(token) for token in tokens]

# 특수 문자 사전 생성: {33: None ...}
# ord(): 아스키 코드 생성
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

# 특수 문자 제거 및 단어 원형 추출
def LemNormalize(text):
    # 텍스트 소문자 변경 후 특수 문자 제거
    text_new = text.lower().translate(remove_punct_dict)
    # 단어 토큰화
    word_tokens = nltk.word_tokenize(text_new)
    # 단어 원형 추출
    return LemTokens(word_tokens)
```

### TF-IDF 수행
```
# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 3000) # max_features = 10000 설정
X = tfidf_vectorizer.fit_transform(df['contents'])

tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_tokens = []
for i in range(len(data)):
    doc = X[i, :].toarray()
    tokens = [tfidf_feature_names[j] for j in np.where(doc > 0)[1]]
    tfidf_tokens.append(tokens)

# 토큰화된 결과를 데이터프레임에 추가
df['tfidf_tokens'] = tfidf_tokens
df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>contents</th>
      <th>tfidf_tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>madrid afp spanish national team coach luis ar...</td>
      <td>[afp, arsenal, coach, comment, decided, face, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bosnia one man hero often another man villain ...</td>
      <td>[citizen, decided, great, hero, lee, look, man...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>yasmine hamdan performs hal also sings film sc...</td>
      <td>[begin, continue, creation, film, living, myst...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>macromedia announced special version contribut...</td>
      <td>[announced, application, creation, designed, e...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>overtheair fix cell phone come qualcomms cdma</td>
      <td>[cell, come, fix, phone]</td>
    </tr>
  </tbody>
</table>
</div>


### LDA 토픽 모델링
이 대회는 카테고리가 정해져 있으므로 문서를 클러스터링 후 각 토픽에 맞게 라벨링해야 한다고 판단했습니다.  
LDA 는 특정 토픽에 대한 단어가 나타날 확률을 나타냅니다.   
즉 문서 안의 단어 분포에 따라 문서마다 특정 토픽이 나타날 확률을 확인하고 가장 높은 확률의 토픽을 best로 설정하여 라벨링 하였습니다.  
LDA 토픽 모델링에 대한 이해는 아래 문서를 참고  
https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/06/01/LDA/

#### 코드 및 시각화
```
import gensim
NUM_TOPICS = 6 # 6개의 토픽
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=30, random_state = 42)
topics = ldamodel.print_topics(num_words=5)
for topic in topics:
    print(topic)
```
(0, '0.016*"new" + 0.012*"service" + 0.010*"microsoft" + 0.010*"internet" + 0.010*"company"')  
(1, '0.014*"game" + 0.011*"night" + 0.011*"year" + 0.010*"win" + 0.009*"time"')  
(2, '0.022*"said" + 0.017*"reuters" + 0.015*"company" + 0.014*"new" + 0.011*"price"')  
(3, '0.025*"said" + 0.014*"reuters" + 0.011*"country" + 0.011*"minister" + 0.011*"official"')  
(4, '0.018*"president" + 0.016*"said" + 0.014*"say" + 0.010*"state" + 0.010*"trump"')  
(5, '0.029*"new" + 0.013*"york" + 0.010*"red" + 0.009*"network" + 0.009*"sport"')  

```
import pyLDAvis.gensim_models

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(vis)
```




