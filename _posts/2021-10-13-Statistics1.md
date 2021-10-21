---
layout: post
title:  "Statistics Study1"
date: 2021-10-13 23:40:00 
category: Statistics
use_math: true

---

## 데이터분석과 통계학
<br>

## 빅데이터의 시대
<br>
이미 '빅데이터의 시대'라고 해도 과언이 아닐만큼 다양한 Smart IT 기기에서 실시간으로 수많은 정보들이 쏟아져나오고 있습니다.<br>
이제는 IT시대에서 DT(Data Technology)시대라는 말이 나올정도 입니다.<br>
앞으로 빅데이터에서 가치를 찾는 것이 기업이 갖춰야 할 역량 중 하나로 떠오를 것입니다.<br>
<br>


조금 더 '**빅데이터**'가 무엇인지 살펴보자면, "기존 데이터베이스 관리도구의 능력을 넘어서는 대량의 정형 또는 심지어 데이터베이스 형태가 아닌 비정형의 데이터 집합조차 포함한 데이터"입니다.<br>
다음 3가지 특성으로 살펴볼 수 있습니다. 
- 데이터의 규모(Volume)
  * 용량의 크기, 데이터의 양이 말그대로 'BIG'

- 데이터 종류의 다양성(Variety)
  * 정렬된 형식의 데이터가 아닌 다양한 포맷의 데이터들

- 데이터가 생산되는 속도(Velocity)
  * 데이터가 생성되고 이동되는 속도가 실시간에 가까움   
    

과거 small data의 경우 타겟이 있고 그에 대한 데이터를 산출해내는 survey, 비교실험이 진행되어, 모집단을 추정하기 위한 표본집단을 분석하는데, 빅데이터의 경우 타겟이 없이 데이터가 산출이 되고, 데이터가 대표성을 띠지 못하고 다소 편향적인 데이터입니다.<br> 그래서 빅데이터 분석분야에서 각 분야의 도메인 지식과 전문성, 창의적 통찰력이 요구되는 것도 그것들을 기반으로 편향적인 데이터를 분석해내기 위함입니다.
<br>

## 왜 통계학인가?
<br>
통계학은 데이터에서 의미를 찾아내는 방법을 다루는 학문입니다. 빅데이터분석기술의 기본이 통계학이라 할 수 있습니다.
<br>

현대 통계학에서는 통계학을 "불확실한 상황에서 효과적인 의사결정을 할 수 있도록 수치자료를 수집하고, 정리하고, 표현하고, 분석하는 이론과 방법을 연구하는 학문"으로 정의하고 있습니다.<br>


<p align = 'center'>
<img src="/assets/2021-10-13-Statistics1/1.jpg" width="40%" height="40%" title="Decision" alt=""/> 
</p>

## 통계학의 두갈래
<br>

<p align = 'center'>
<img src="/assets/2021-10-13-Statistics1/2.jpg" width="40%" height="40%" title="Decision" alt=""/> 
</p>

### 기술통계(Descriptive Statistics)
측정이나 실험을 통해 수집한 통계자료의 정리/표현/요약/해석을 통하여 자료특성을 규명하는 방법과 기법으로, 크게 데이터의 집중화 경향에 대한 기법과 데이터의 흩어짐 정도에 대한 기법으로 나눌 수 있습니다.<br>

- 기술통계치: count, mean, standard dev, min, 1Q, median, 3Q, max 등의 데이터를 설명 하는 값(혹은 통계치)들

<p align = 'center'>
<img src="/assets/2021-10-13-Statistics1/3.jpg" width="40%" height="40%" title="Decision" alt=""/> 
</p>

### 추리통계(Inferential Statistics)
한 모집단에서 추출한 표본에 대해 기술통계학을 이용하여 구한 표본정보에 입각하여 그의 모집단의 어떤 특성에 대해 결론을 추론하는 절차와 기법으로, 크게 연역법과 귀납법을 활용합니다.<br>
**빅데이터분석에서 더더욱 중요시되는 것이 "추리통계"입니다. 의미가 있는 가치를 찾기 위해 가설을 세우고 검증하는 순환이 필요하기 때문입니다.**<br>



## 가설검정
<br>

통계적 가설검정은 "귀무가설(Null hypothesis: H0)"과 대립가설(Alternative hypothesis: H1)을 설정하고, 추출한 데이터를 이용하여 얻은 어떤 관찰값(검정통계량)을 기반으로 어느 가설을 채택할 지 결정하는 것을 말합니다.<br>


통계학에서의 가설은 어디까지나 어떤 모집단의 모수에 대한 잠정적인 추론입니다.<br> 따라서 귀무가설은 "모집단의 특성에 대해 옳다고 제안하는 잠정적인 주장"이고, 보통 '차이가 없다', '-의 효과가 없다', '-와 같다'로 표현됩니다.<br>반면 대립가설은 귀무가설이 거짓이라면 대안적으로 참이 되는 가설입니다.<br>

두 가설중에서 어떤 것을 채택하든, 표본데이터를 가지고 모집단을 추론하기 때문에 오류의 가능성이 존재합니다.<br>
통계학에서는 '1종 오류', '2종 오류'가 존재하며, 이 통계적 오류들의 최소한 허용범위를 설정하여 가설검정을 합니다.<br>

- 1종오류란, 귀무가설이 참인데 기각하는 경우로, 효과가 없음에도 불구하고 효과가 있다고 판단하는 오류입니다. (오류 확률 : *a* )

- 2종오류란, 귀무가설이 거짓인데 채택하는 경우로, 효과가 있음에도 불구하고 효과가 없다고 판단하는 오류입니다. (오류 확률 : *b*)

가설검정에서는 1종 오류만 고려하며, 1종오류의 가능성을 보통 1%, 5%로 임계값을 설정하고, 귀무가설을 채택하거나 기각합니다.<br>
이 떄 임계값과 비교되는 값을 p-value라 하고, 1종 오류가 발생해도 감안할 수 있는 수준을 '유의수준'이라고 합니다. <br>

조금 더 정리하자면, p-value가 5%이하라면, 귀무가설이 틀렸지만 우연히 맞을 확률이 5%이하로, 귀무가설을 기각한다는 판별을 하게 됩니다. <br>

## 검정통계량
<br>

검정통계량이란, 수집한 데이터를 이용해서 계산한 "확률변수"이고, 이 변수는 확률함수를 이용해, 표본 통계량이 발생할 확률을 계산할 수 있습니다.<br>
 검정통계량을 통해 계산된 확률이 바로 p-value입니다.

검정통계량은 통계기법이 사용하는 확률 분포함수에 따라 Z(정규분포), t (t-분포), F( F- 분포), X^2 (카이제곱 분포) 등이 사용될 수 있고, 이에 상응되는 p-value가 계산이 가능합니다.<br>

각 검정통계량은 만족되는 조건에 따라 활용이 결정되는데, 특히 t-test의 경우 표본 크기가 작고, 모집단 표준편차를 알 수 없는 경우에 두 모집단의 평균이 서로 다른지 여부를 비교하고 분석하는 데 사용되는 통계 검정으로 이해할 수 있습니다.<br>

일반적으로 고등학교에서 배웠던 Z-test의 경우 표본 평균분포가 정규분포를 따르고, 모집단의 표준편차를 알고 있거나 알려진 것을 가정하여 test를 진행합니다.<br>

## t-test
<br>

### One-sample t-test
one-sample t-test는 표본집단의 데이터를 가지고 모집단의 평균이 특정 값과 같은지 검정하는 통계방법입니다.<br> 이때 우리는 모집단의 평균을 알아야만이 가능한 통계방법입니다. 

### Paired t-test
paired t-test는 짝을 이룬 두 변수간 차이의 평균이 0인지에 대한 검정입니다.<br>
one sample t-test의 특수한 경우로, 특징은 서로 연관성이 있는 두 대상으로부터 측정된 값이어서 비교하는 두 변수들 사이에 상관관계가 존재한다는 것입니다. <br>
예시로는, 10명의 환자에게 수면제를 투여한 후의 수면시간 증가에 대한 paired t-test가 대표적입니다. <br>

### Two-sample t-test
two-sample t-test는 서로 독립적인 두 집단의 평균 차이가 0인지에 대한 검정으로, 두 집단의 평균이 같은지를 비교하여 개입효과의 차이를 평가하는 것입니다.<br>
유의할 점은 두 집단의 분산이 같은지 다른지에 따라 방법이 달라진다는 것입니다.<br>
예시로는 실험군과 대조군에 서로 다른 처리를 한 후 두 집단의 평균이 같은지를 비교하여 그 처리의 효과를 평가하는, 대부분의 실험 연구가 이 two sample t-test에 해당합니다.<br>



```python
#one-sample t-test
import pandas as pd
import numpy as np
from scipy import stats
trees = pd.read_csv('https://ds-lecture-data.s3.ap-northeast-2.amazonaws.com/seoul_tree/seoul_tree.txt', sep = '\t', skiprows = 1)
trees = trees.replace({'-':0})
tree1 = pd.to_numeric(trees['은행나무'].str.replace(',','')) # 은행나무 데이터
tree2 = pd.to_numeric(trees['느티나무'].str.replace(',','')) # 느티나무 데이터

print("진짜 평균은 {}입니다.".format(np.mean(tree1)))
print("p-value는 귀무가설을 채택할 확률로 생각해도됨")

print("평균이 20000이랑 같니?", stats.ttest_1samp(tree1, 20000))
print("평균이 8000이랑 같니?",stats.ttest_1samp(tree1, 8000))
print("평균이 7800이랑 같니?", stats.ttest_1samp(tree1, 7800))
print("평균이 5000이랑 같니?",stats.ttest_1samp(tree1, 5000))
print("평균이 4000이랑 같니?", stats.ttest_1samp(tree1, 4000))
```

    진짜 평균은 7717.857142857143입니다.
    p-value는 귀무가설을 채택할 확률로 생각해도됨
    평균이 20000이랑 같니? Ttest_1sampResult(statistic=-3.2845108721916954, pvalue=0.0028293224983187673)
    평균이 8000이랑 같니? Ttest_1sampResult(statistic=-0.07545110756125144, pvalue=0.9404120803789722)
    평균이 7800이랑 같니? Ttest_1sampResult(statistic=-0.02196677815074404, pvalue=0.98263599594694)
    평균이 5000이랑 같니? Ttest_1sampResult(statistic=0.7268138335963595, pvalue=0.473594950422093)
    평균이 4000이랑 같니? Ttest_1sampResult(statistic=0.9942354806488966, pvalue=0.3289356132079231)
    


```python
print("은행나무 평균은 {}입니다.".format(np.mean(tree1)))
print("느티나무 평균은 {}입니다.".format(np.mean(tree2)))
print("두 집단의 평균이 같니?", stats.ttest_ind(tree1,tree2))

```

    은행나무 평균은 7717.857142857143입니다.
    느티나무 평균은 2676.6428571428573입니다.
    두 집단의 평균이 같니? Ttest_indResult(statistic=1.2730451277184196, pvalue=0.20845547747699233)
    

## 신뢰구간
<br>
신뢰구간은 모수가 실제로 포함될 것으로 예측되는 범위입니다.<br>
즉 신뢰도가 95% 라는 의미는 표본을 100번 뽑았을때 95번은 신뢰구간 내에 모집단의 평균이 포함된다는 의미입니다.   

<p align = 'center'>
<img src="/assets/2021-10-13-Statistics1/4.png" width="35%" height="35%" title="Decision" alt=""/> 
</p>

## Reference
<br>
http://www.databaser.net/moniwiki/wiki.php/%ED%86%B5%EA%B3%84%ED%95%99%EC%9D%B4%EB%9E%80%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80#s-4<br>
https://drhongdatanote.tistory.com/77?category=648822<br>
https://m.blog.naver.com/sendmethere/221333164258<br>
https://drhongdatanote.tistory.com/80?category=648822<br>
https://m.blog.naver.com/sw4r/222035001670<br>
https://angeloyeo.github.io/2021/01/05/confidence_interval.html<br>
