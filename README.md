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
![image](https://github.com/Junoflows/NPL_Clustering_LDA/assets/108385417/0ced5e8d-6b26-4671-9255-25d595058820)



+ 이 토픽별 단어 중요도를 확인한 후 직접 라벨링 해주었습니다.  
Business, Entertainment, Politics, Sports, Tech, World  
Politics - 4 <br/>
Tech - 0 <br/>
Sports - 1 <br/>
Business - 2 <br/>
World - 3 <br/>
Entertainment - 5 <br/>

+ 문서별 토픽 비율과 best 토픽을 열로 추가하였습니다.
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num</th>
      <th>best_topic</th>
      <th>best_topic_rate</th>
      <th>topic_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0.3871</td>
      <td>[(3, 0.38706496), (1, 0.30021867), (4, 0.16776...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0.7737</td>
      <td>[(1, 0.77370757), (3, 0.1520915), (0, 0.018572...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>0.5535</td>
      <td>[(0, 0.55345213), (1, 0.2692915), (5, 0.131465...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>0.7849</td>
      <td>[(0, 0.7848793), (2, 0.14836921), (4, 0.016694...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>0.8331</td>
      <td>[(0, 0.8331153), (1, 0.033446807), (4, 0.03343...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>59995</th>
      <td>59995</td>
      <td>1</td>
      <td>0.5471</td>
      <td>[(1, 0.5470573), (4, 0.38566053), (3, 0.016893...</td>
    </tr>
    <tr>
      <th>59996</th>
      <td>59996</td>
      <td>2</td>
      <td>0.5543</td>
      <td>[(2, 0.55431587), (3, 0.29467893), (0, 0.11966...</td>
    </tr>
    <tr>
      <th>59997</th>
      <td>59997</td>
      <td>1</td>
      <td>0.4721</td>
      <td>[(1, 0.47208622), (0, 0.29982916), (5, 0.18620...</td>
    </tr>
    <tr>
      <th>59998</th>
      <td>59998</td>
      <td>2</td>
      <td>0.8808</td>
      <td>[(2, 0.88078445), (4, 0.023868008), (0, 0.0238...</td>
    </tr>
    <tr>
      <th>59999</th>
      <td>59999</td>
      <td>5</td>
      <td>0.7618</td>
      <td>[(5, 0.76178455), (2, 0.1047389), (1, 0.097624...</td>
    </tr>
  </tbody>
</table>
<p>60000 rows × 4 columns</p>
</div>

+ 원래 데이터에 best 토픽 추가하고 카테고리 이름을 주어진 형식으로 재설정해주었습니다.  
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>contents</th>
      <th>tfidf_tokens</th>
      <th>best_topic_rate</th>
      <th>topic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>madrid afp spanish national team coach luis ar...</td>
      <td>[afp, arsenal, coach, comment, decided, face, ...</td>
      <td>0.3871</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bosnia one man hero often another man villain ...</td>
      <td>[citizen, decided, great, hero, lee, look, man...</td>
      <td>0.7737</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>yasmine hamdan performs hal also sings film sc...</td>
      <td>[begin, continue, creation, film, living, myst...</td>
      <td>0.5535</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>macromedia announced special version contribut...</td>
      <td>[announced, application, creation, designed, e...</td>
      <td>0.7849</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>overtheair fix cell phone come qualcomms cdma</td>
      <td>[cell, come, fix, phone]</td>
      <td>0.8331</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


## 결과 및 리뷰
+ 대학교 전공 수업에서 LDA에 대해 깊게 배우게 되어 군집분류에 사용해보면 좋을 것 같다는 생각으로 LDA 토픽 모델링을 군집분류에 사용해보았다.
+ 위 그래프에서 군집화가 6개로 잘 되어있다고 생각했는데 결과는 좋지 않았고, 이 후 코드공유에서 높은 성능의 분석방법은 Kmeans 나 TSNE 등의 군집분석을 사용했었다.
+ LDA의 분류 성능이 좋지 못한 이유에 대해 고찰
  + Kmeans 는 하나의 문서가 하나의 토픽이라는 가정을 하지만, LDA는 한 문서에서 여러 종류의 토픽이 존재하고 그 중 비율이 가장 높은 것을 선택했다.
  + 뉴스 기사 분류는 주로 한 문서에 하나의 토픽이 존재하기 때문에 Kmeans 가 더 좋게 결과가 나온 것으로 판단된다.
  + LDA가 분류 성능이 항상 나쁜 것은 아니지만 뉴스 기사 분류 같은 상황에서는 Kmeans 를 사용하는게 적절하다.
