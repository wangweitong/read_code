
# COMPREHENSIVE DATA EXPLORATION WITH PYTHON
[Pedro Marcelino](http://pmarcelino.com) - February 2017

Other Kernels: [Data analysis and feature extraction with Python
](https://www.kaggle.com/pmarcelino/data-analysis-and-feature-extraction-with-python)

----------

<b>'The most difficult thing in life is to know yourself'</b>

This quote belongs to Thales of Miletus. Thales was a Greek/Phonecian philosopher, mathematician and astronomer, which is recognised as the first individual in Western civilisation known to have entertained and engaged in scientific thought (source: https://en.wikipedia.org/wiki/Thales)

I wouldn't say that knowing your data is the most difficult thing in data science, but it is time-consuming. Therefore, it's easy to overlook this initial step and jump too soon into the water.

So I tried to learn how to swim before jumping into the water. Based on [Hair et al. (2013)](https://amzn.to/2JuDmvo), chapter 'Examining your data', I did my best to follow a comprehensive, but not exhaustive, analysis of the data. I'm far from reporting a rigorous study in this kernel, but I hope that it can be useful for the community, so I'm sharing how I applied some of those data analysis principles to this problem.

Despite the strange names I gave to the chapters, what we are doing in this kernel is something like:

1. <b>Understand the problem</b>. We'll look at each variable and do a philosophical analysis about their meaning and importance for this problem.
2. <b>Univariable study</b>. We'll just focus on the dependent variable ('SalePrice') and try to know a little bit more about it.
3. <b>Multivariate study</b>. We'll try to understand how the dependent variable and independent variables relate.
4. <b>Basic cleaning</b>. We'll clean the dataset and handle the missing data, outliers and categorical variables.
5. <b>Test assumptions</b>. We'll check if our data meets the assumptions required by most multivariate techniques.

Now, it's time to have fun!

<b>'生活中最困难的事情就是了解自己'</b>

这句话属于米利都的泰勒斯。泰勒斯是希腊/电话的哲学家，数学家和天文学家，被公认为西方文明中第一个接受和参与科学思想的人（来源：https：//en.wikipedia.org/wiki/Thales）

我不是说知道你的数据是数据科学中最困难的事情，但它很耗时。因此，很容易忽视这个初始步骤，并且很快就会跳入水中。

所以我试着在跳入水中之前学会游泳。基于[Hair et al。 （2013）]（https://amzn.to/2JuDmvo），“检查您的数据”一章，我尽力遵循全面但非详尽的数据分析。我还没有在这个内核中报告严格的研究，但我希望它对社区有用，所以我分享了我如何将这些数据分析原则应用于这个问题。

尽管我给这些章节写了一些奇怪的名字，但我们在这个内核中做的是：

1. <b>了解问题</b>。我们将查看每个变量，并对其对此问题的意义和重要性进行哲学分析。
2. <b>单变量研究</b>。我们只关注因变量（'SalePrice'）并尝试更多地了解它。
3. <b>多变量研究</b>。我们将尝试理解因变量和自变量如何相关。
4. <b>基本清洁</b>。我们将清理数据集并处理缺失的数据，异常值和分类变量。
5. <b>测试假设</b>。我们将检查我们的数据是否符合大多数多变量技术所需的假设。

现在，是时候玩得开心了！


```python
#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```


```python
#bring in the six packs
df_train = pd.read_csv('../input/train.csv')
```


```python
#check the decoration
df_train.columns
```

# 1. So... What can we expect?

In order to understand our data, we can look at each variable and try to understand their meaning and relevance to this problem. I know this is time-consuming, but it will give us the flavour of our dataset.

In order to have some discipline in our analysis, we can create an Excel spreadsheet with the following columns:
* <b>Variable</b> - Variable name.
* <b>Type</b> - Identification of the variables' type. There are two possible values for this field: 'numerical' or 'categorical'. By 'numerical' we mean variables for which the values are numbers, and by 'categorical' we mean variables for which the values are categories.
* <b>Segment</b> - Identification of the variables' segment. We can define three possible segments: building, space or location. When we say 'building', we mean a variable that relates to the physical characteristics of the building (e.g. 'OverallQual'). When we say 'space', we mean a variable that reports space properties of the house (e.g. 'TotalBsmtSF'). Finally, when we say a 'location', we mean a variable that gives information about the place where the house is located (e.g. 'Neighborhood').
* <b>Expectation</b> - Our expectation about the variable influence in 'SalePrice'. We can use a categorical scale with 'High', 'Medium' and 'Low' as possible values.
* <b>Conclusion</b> - Our conclusions about the importance of the variable, after we give a quick look at the data. We can keep with the same categorical scale as in 'Expectation'.
* <b>Comments</b> - Any general comments that occured to us.

While 'Type' and 'Segment' is just for possible future reference, the column 'Expectation' is important because it will help us develop a 'sixth sense'. To fill this column, we should read the description of all the variables and, one by one, ask ourselves:

* Do we think about this variable when we are buying a house? (e.g. When we think about the house of our dreams, do we care about its 'Masonry veneer type'?).
* If so, how important would this variable be? (e.g. What is the impact of having 'Excellent' material on the exterior instead of 'Poor'? And of having 'Excellent' instead of 'Good'?).
* Is this information already described in any other variable? (e.g. If 'LandContour' gives the flatness of the property, do we really need to know the 'LandSlope'?).

After this daunting exercise, we can filter the spreadsheet and look carefully to the variables with 'High' 'Expectation'. Then, we can rush into some scatter plots between those variables and 'SalePrice', filling in the 'Conclusion' column which is just the correction of our expectations.

I went through this process and concluded that the following variables can play an important role in this problem:

* OverallQual (which is a variable that I don't like because I don't know how it was computed; a funny exercise would be to predict 'OverallQual' using all the other variables available).
* YearBuilt.
* TotalBsmtSF.
* GrLivArea.

I ended up with two 'building' variables ('OverallQual' and 'YearBuilt') and two 'space' variables ('TotalBsmtSF' and 'GrLivArea'). This might be a little bit unexpected as it goes against the real estate mantra that all that matters is 'location, location and location'. It is possible that this quick data examination process was a bit harsh for categorical variables. For example, I expected the 'Neigborhood' variable to be more relevant, but after the data examination I ended up excluding it. Maybe this is related to the use of scatter plots instead of boxplots, which are more suitable for categorical variables visualization. The way we visualize data often influences our conclusions.

However, the main point of this exercise was to think a little about our data and expectactions, so I think we achieved our goal. Now it's time for 'a little less conversation, a little more action please'. Let's <b>shake it!</b>

# 那么......我们能期待什么？
为了理解我们的数据，我们可以查看每个变量并尝试理解它们的含义以及与此问题的相关性。我知道这很费时间，但它会给我们带来数据集的味道。

为了在我们的分析中有一些规则，我们可以创建一个包含以下列的Excel电子表格：

变量 - 变量名称。
类型 - 变量类型的标识。此字段有两个可能的值：“数字”或“分类”。 “数字”是指值为数字的变量，而“分类”是指值为类别的变量。
细分 - 变量细分的识别。我们可以定义三个可能的部分：建筑，空间或位置。当我们说'建筑'时，我们指的是与建筑物的物理特征相关的变量（例如'OverallQual'）。当我们说'空间'时，我们指的是一个报告房屋空间属性的变量（例如'TotalBsmtSF'）。最后，当我们说“位置”时，我们指的是一个变量，它提供有关房屋所在位置的信息（例如“邻居”）。
期望 - 我们对'SalePrice'变量影响的期望。我们可以使用“高”，“中”和“低”作为可能值的分类量表。
结论 - 在我们快速查看数据之后，我们对变量重要性的结论。我们可以保持与“期望”中相同的分类标度。
评论 - 发生在我们身上的任何一般性评论。
虽然“类型”和“细分”仅供将来参考，但“期望”一栏很重要，因为它将帮助我们发展“第六感”。要填写此专栏，我们应该阅读所有变量的描述，并逐一问自己：

我们买房子的时候会考虑这个变量吗？ （例如，当我们想到我们梦想中的房子时，我们是否关心它的'砌体贴面类型'？）。
如果是这样，这个变量有多重要？ （例如，在外部使用'优质'材料而不是'差'会产生什么影响？并且'优秀'而不是'好'会有什么影响？）。
这些信息是否已在任何其他变量中描述过？ （例如，如果'LandContour'给出了房产的平坦度，我们真的需要知道'LandSlope'吗？）。
在这个令人生畏的练习之后，我们可以过滤电子表格并仔细查看具有“高”'期望'的变量。然后，我们可以匆匆进入这些变量和'SalePrice'之间的一些散点图，填写'结论'栏，这只是我们期望的修正。

我经历了这个过程并得出结论，以下变量可以在这个问题中发挥重要作用：

总体质量（这是一个我不喜欢的变量，因为我不知道它是如何计算的;一个有趣的练习是使用所有其他可用的变量来预测'总体质量'）。
YearBuilt。
TotalBsmtSF。
GrLivArea。
我最终得到了两个“构建​​”变量（'OverallQual'和'YearBuilt'）和两个'space'变量（'TotalBsmtSF'和'GrLivArea'）。这可能有点出乎意料，因为它违背了房地产的口头禅，所有重要的是“位置，位置和位置”。对于分类变量，这种快速数据检查过程可能有点苛刻。例如，我预计'Neigborhood'变量更具相关性，但在数据检查后我最终将其排除在外。也许这与使用散点图而不是箱图有关，这更适合于分类变量可视化。我们可视化数据的方式通常会影响我们的结论。

然而，这个练习的主要目的是想一想我们的数据和期望，所以我认为我们实现了目标。现在是时候“少谈一点，请多一点动作”。让我们动摇吧！

# 2. First things first: analysing 'SalePrice'

'SalePrice' is the reason of our quest. It's like when we're going to a party. We always have a reason to be there. Usually, women are that reason. (disclaimer: adapt it to men, dancing or alcohol, according to your preferences)

Using the women analogy, let's build a little story, the story of 'How we met 'SalePrice''.

*Everything started in our Kaggle party, when we were looking for a dance partner. After a while searching in the dance floor, we saw a girl, near the bar, using dance shoes. That's a sign that she's there to dance. We spend much time doing predictive modelling and participating in analytics competitions, so talking with girls is not one of our super powers. Even so, we gave it a try:*

*'Hi, I'm Kaggly! And you? 'SalePrice'? What a beautiful name! You know 'SalePrice', could you give me some data about you? I just developed a model to calculate the probability of a successful relationship between two people. I'd like to apply it to us!'*

# 首先要做的事情是：分析'SalePrice'
'SalePrice'是我们追求的原因。就像我们去参加派对一样。我们总是有理由去那里。通常，女性就是这个原因。 （免责声明：根据您的喜好，适应男士，舞蹈或酒精）

使用女性比喻，让我们构建一个小故事，“我们如何遇见'SalePrice'的故事。

当我们寻找舞伴时，一切都始于我们的Kaggle派对。经过一段时间在舞池里搜索，我们看到一个女孩，在酒吧附近，使用舞鞋。这是她在那里跳舞的标志。我们花了很多时间进行预测建模并参与分析比赛，因此与女孩交谈并不是我们的超能力之一。即便如此，我们试了一下：

“嗨，我是Kaggly！你呢？ '销售价格'？多么美丽的名字！你知道'SalePrice'，你能给我一些关于你的数据吗？我刚开发了一个模型来计算两者之间成功关系的概率。我想把它应用到我们这里！“


```python
#descriptive statistics summary
#描述性统计
df_train['SalePrice'].describe()
```

*'Very well... It seems that your minimum price is larger than zero. Excellent! You don't have one of those personal traits that would destroy my model! Do you have any picture that you can send me? I don't know... like, you in the beach... or maybe a selfie in the gym?'*
*'很好......似乎你的最低价格大于零。 优秀！ 你没有那些破坏我模特的个人特征！ 你有任何可以寄给我的照片吗？ 我不知道......就像，你在沙滩上......或者在健身房拍照？'*



```python
#histogram
#直方图
sns.distplot(df_train['SalePrice']);
```

*'Ah! I see you that you use seaborn makeup when you're going out... That's so elegant! I also see that you:*

* *<b>Deviate from the normal distribution.</b>*
* *<b>Have appreciable positive skewness.</b>*
* *<b>Show peakedness.</b>*

*This is getting interesting! 'SalePrice', could you give me your body measures?'*

*'啊！ 我告诉你，当你外出时，你会使用seaborn化妆......那太优雅了！ 我也看到了你：*

* * <b>偏离正常分布。</b> *
* * <b>有明显的正面偏斜。</b> *
* * <b>显示达到顶峰。</b> *

*这很有意思！ 'SalePrice'，你能告诉我你的身体测量吗？'*


```python
#skewness and kurtosis
# 偏度(skewness)和峰度(kurtosis)：
#   偏度能够反应分布的对称情况，右偏（也叫正偏），在图像上表现为数据右边脱了一个长长的尾巴，这时大多数值分布在左侧，有一小部分值分布在右侧。
#   峰度反应的是图像的尖锐程度：峰度越大，表现在图像上面是中心点越尖锐。在相同方差的情况下，中间一大部分的值方差都很小，为了达到和正太分布方差相同的目的，必须有一些值离中心点越远，所以这就是所说的“厚尾”，反应的是异常点增多这一现象。

print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
```

![](http://)*'Amazing! If my love calculator is correct, our success probability is 97.834657%. I think we should meet again! Please, keep my number and give me a call if you're free next Friday. See you in a while, crocodile!'*

*'惊人！ 如果我的爱情计算器是正确的，我们的成功概率是97.834657％。 我想我们应该再见面！ 如果你下周五有空，请保留我的电话号码给我打个电话。 有一段时间见到你，鳄鱼！'*

1. # 'SalePrice', her buddies and her interests

*It is military wisdom to choose the terrain where you will fight. As soon as 'SalePrice' walked away, we went to Facebook. Yes, now this is getting serious. Notice that this is not stalking. It's just an intense research of an individual, if you know what I mean.*

*According to her profile, we have some common friends. Besides Chuck Norris, we both know 'GrLivArea' and 'TotalBsmtSF'. Moreover, we also have common interests such as 'OverallQual' and 'YearBuilt'. This looks promising!*

*To take the most out of our research, we will start by looking carefully at the profiles of our common friends and later we will focus on our common interests.*

# 'SalePrice'，她的朋友和她的兴趣
*选择你将要战斗的地形是军事智慧。 一旦'SalePrice'离开，我们就去了Facebook。 是的，现在情况变得严肃了。 请注意，这不是跟踪。 如果你知道我的意思，那只是对个人的激烈研究。*

*根据她的个人资料，我们有一些共同的朋友。 除了查克诺里斯，我们都知道'GrLivArea'和'TotalBsmtSF'。 此外，我们也有共同的兴趣，如'OverallQual'和'YearBuilt'。 这看起来很有希望！*

*为了充分利用我们的研究，我们将首先仔细研究我们共同朋友的概况，然后我们将关注我们的共同兴趣。*

### Relationship with numerical variables


```python
#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
```

*Hmmm... It seems that 'SalePrice' and 'GrLivArea' are really old friends, with a <b>linear relationship.</b>*

*And what about 'TotalBsmtSF'?*

*嗯......似乎'SalePrice'和'GrLivArea'真的是老朋友，有<b>线性关系。</b> *

*那么'TotalBsmtSF'呢？*


```python
#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
```

*'TotalBsmtSF' is also a great friend of 'SalePrice' but this seems a much more emotional relationship! Everything is ok and suddenly, in a <b>strong linear (exponential?)</b> reaction, everything changes. Moreover, it's clear that sometimes 'TotalBsmtSF' closes in itself and gives zero credit to 'SalePrice'.*

*'TotalBsmtSF'也是'SalePrice'的好朋友，但这似乎是一种更加情感化的关系！ 一切都很好，突然间，在<b>强烈的线性（指数？）</b>反应中，一切都在变化。 此外，很明显，有时候“TotalBsmtSF”本身会关闭，并且对“SalePrice”给予零信誉。*

### Relationship with categorical features

## 明确的特征之间的关系


```python
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
```

1. *Like all the pretty girls, 'SalePrice' enjoys 'OverallQual'. Note to self: consider whether McDonald's is suitable for the first date.*

1. *像所有漂亮的女孩一样，'SalePrice'喜欢'OverallQual'。 自我注意：考虑麦当劳是否适合第一次约会。*


```python
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
```

*Although it's not a strong tendency, I'd say that 'SalePrice' is more prone to spend more money in new stuff than in old relics.*

<b>Note</b>: we don't know if 'SalePrice' is in constant prices. Constant prices try to remove the effect of inflation. If 'SalePrice' is not in constant prices, it should be, so than prices are comparable over the years.

*虽然这不是一个强烈的倾向，但我会说'SalePrice'更倾向于花更多钱买新东西而不是旧文物。*

<b>注意</b>：我们不知道'SalePrice'是否价格不变。 不变价格试图消除通胀的影响。 如果'SalePrice'价格不是固定的，那么它应该是多年来价格相当的。

### In summary

Stories aside, we can conclude that:

* 'GrLivArea' and 'TotalBsmtSF' seem to be linearly related with 'SalePrice'. Both relationships are positive, which means that as one variable increases, the other also increases. In the case of 'TotalBsmtSF', we can see that the slope of the linear relationship is particularly high.
* 'OverallQual' and 'YearBuilt' also seem to be related with 'SalePrice'. The relationship seems to be stronger in the case of 'OverallQual', where the box plot shows how sales prices increase with the overall quality.

We just analysed four variables, but there are many other that we should analyse. The trick here seems to be the choice of the right features (feature selection) and not the definition of complex relationships between them (feature engineering).

That said, let's separate the wheat from the chaff.

### 综上所述

抛开故事，我们可以得出结论：

*'GrLivArea'和'TotalBsmtSF'似乎与'SalePrice'线性相关。 两种关系都是正的，这意味着当一个变量增加时，另一个变量增加。 在'TotalBsmtSF'的情况下，我们可以看到线性关系的斜率特别高。
*'OverallQual'和'YearBuilt'似乎也与'SalePrice'有关。 在'OverallQual'的情况下，这种关系似乎更强，其中箱形图显示销售价格如何随着整体质量而增加。

我们刚刚分析了四个变量，但还有许多其他我们应该分析的变量。 这里的诀窍似乎是选择正确的特征（特征选择）而不是它们之间复杂关系的定义（特征工程）。

那就是说，让我们将小麦与谷壳分开。

# 3. Keep calm and work smart

# 3.保持冷静，聪明地工作

Until now we just followed our intuition and analysed the variables we thought were important. In spite of our efforts to give an objective character to our analysis, we must say that our starting point was subjective. 

As an engineer, I don't feel comfortable with this approach. All my education was about developing a disciplined mind, able to withstand the winds of subjectivity. There's a reason for that. Try to be subjective in structural engineering and you will see physics making things fall down. It can hurt.

So, let's overcome inertia and do a more objective analysis.

到现在为止，我们只是按照我们的直觉来分析我们认为重要的变量。 尽管我们努力为我们的分析提供客观的特征，但我们必须说我们的出发点是主观的。

作为一名工程师，我对这种方法感到不舒服。 我所有的教育都是关于培养一种能够抵御风险的纪律思维。 这是有原因的。 尝试在结构工程中主观，你会觉得违背常理。

所以，让我们克服惯性，做一个更客观的分析。

### The 'plasma soup'

'In the very beginning there was nothing except for a plasma soup. What is known of these brief moments in time, at the start of our study of cosmology, is largely conjectural. However, science has devised some sketch of what probably happened, based on what is known about the universe today.' (source: http://umich.edu/~gs265/bigbang.htm) 

To explore the universe, we will start with some practical recipes to make sense of our 'plasma soup':
* Correlation matrix (heatmap style).
* 'SalePrice' correlation matrix (zoomed heatmap style).
* Scatter plots between the most correlated variables (move like Jagger style).

### '等离子汤'

“一开始除了等离子汤外什么都没有。 在我们对宇宙学的研究开始时，对这些短暂时刻的了解在很大程度上是推测的。 然而，科学已经根据今天对宇宙的了解，设计了一些可能发生的概述。 （来源：http：//umich.edu/~gs265/bigbang.htm）

为了探索宇宙，我们将从一些实用的食谱开始，以了解我们的'等离子汤'：
*相关矩阵（热图样式）。
*'SalePrice'相关矩阵（缩放热图样式）。
*最相关变量之间的散点图（像Jagger样式一样移动）。

#### Correlation matrix (heatmap style)
#### 相关矩阵（热图样式）


```python
#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
```

In my opinion, this heatmap is the best way to get a quick overview of our 'plasma soup' and its relationships. (Thank you @seaborn!)

At first sight, there are two red colored squares that get my attention. The first one refers to the 'TotalBsmtSF' and '1stFlrSF' variables, and the second one refers to the 'Garage*X*' variables. Both cases show how significant the correlation is between these variables. Actually, this correlation is so strong that it can indicate a situation of multicollinearity. If we think about these variables, we can conclude that they give almost the same information so multicollinearity really occurs. Heatmaps are great to detect this kind of situations and in problems dominated by feature selection, like ours, they are an essential tool.

Another thing that got my attention was the 'SalePrice' correlations. We can see our well-known 'GrLivArea', 'TotalBsmtSF', and 'OverallQual' saying a big 'Hi!', but we can also see many other variables that should be taken into account. That's what we will do next.

在我看来，这个热图是快速了解我们的'血浆汤'及其关系的最佳方式。 （谢谢@seaborn！）
看来是颜色越浅关系性越强

乍一看，有两个红色方块引起了我的注意。第一个引用'TotalBsmtSF'和'1stFlrSF'变量，第二个引用'Garage * X *'变量。两种情况都表明这些变量之间的相关性有多大。实际上，这种相关性非常强，可以表明多重共线性的情况。如果我们考虑这些变量，我们可以得出结论，它们给出了几乎相同的信息，因此实际上发生了多重共线性。热图非常适合检测这种情况，并且在功能选择主导的问题中，如我们的，它们是必不可少的工具。

引起我注意的另一件事是'SalePrice'相关性。我们可以看到我们众所周知的'GrLivArea'，'TotalBsmtSF'和'OverallQual'说一个大'嗨！'，但我们也可以看到许多其他应该考虑的变量。这就是我们接下来要做的事情。

#### 'SalePrice' correlation matrix (zoomed heatmap style)


```python
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
```

According to our crystal ball, these are the variables most correlated with 'SalePrice'. My thoughts on this:

* 'OverallQual', 'GrLivArea' and 'TotalBsmtSF' are strongly correlated with 'SalePrice'. Check!
* 'GarageCars' and 'GarageArea' are also some of the most strongly correlated variables. However, as we discussed in the last sub-point, the number of cars that fit into the garage is a consequence of the garage area. 'GarageCars' and 'GarageArea' are like twin brothers. You'll never be able to distinguish them. Therefore, we just need one of these variables in our analysis (we can keep 'GarageCars' since its correlation with 'SalePrice' is higher).
* 'TotalBsmtSF' and '1stFloor' also seem to be twin brothers. We can keep 'TotalBsmtSF' just to say that our first guess was right (re-read 'So... What can we expect?').
* 'FullBath'?? Really? 
* 'TotRmsAbvGrd' and 'GrLivArea', twin brothers again. Is this dataset from Chernobyl?
* Ah... 'YearBuilt'... It seems that 'YearBuilt' is slightly correlated with 'SalePrice'. Honestly, it scares me to think about 'YearBuilt' because I start feeling that we should do a little bit of time-series analysis to get this right. I'll leave this as a homework for you.

Let's proceed to the scatter plots.

根据我们的水晶球，这些是与'SalePrice'最相关的变量。我对此的看法：

*'OverallQual'，'GrLivArea'和'TotalBsmtSF'与'SalePrice'密切相关。校验！
*'GarageCars'和'GarageArea'也是一些最强相关的变量。然而，正如我们在上一个子点中所讨论的那样，车库中的车辆数量是车库区域的结果。 'GarageCars'和'GarageArea'就像孪生兄弟。你永远无法区分它们。因此，我们在分析中只需要其中一个变量（我们可以保留'GarageCars'，因为它与'SalePrice'的相关性更高）。
*'TotalBsmtSF'和'1stFloor'似乎也是孪生兄弟。我们可以保持'TotalBsmtSF'只是说我们的第一个猜测是正确的（重新阅读'那么......我们能期待什么？'）。
*'FullBath'??真？
*'TotRmsAbvGrd'和'GrLivArea'，孪生兄弟。这个数据集来自切尔诺贝利吗？
*啊......'YearBuilt'......似乎'YearBuilt'与'SalePrice'略有关联。老实说，让我想起'YearBuilt'让我感到害怕，因为我开始觉得我们应该做一些时间序列分析才能做到这一点。我会留下这个作为你的功课。

我们来看散点图。

#### Scatter plots between 'SalePrice' and correlated variables (move like Jagger style)
#### SalePrice和相关变量之间的散点图（像Jagger样式一样移动）

Get ready for what you're about to see. I must confess that the first time I saw these scatter plots I was totally blown away! So much information in so short space... It's just amazing. Once more, thank you @seaborn! You make me 'move like Jagger'!

为即将看到的内容做好准备。 我必须承认，我第一次看到这些散点图时，我完全被吹走了！ 这么短的空间里有如此多的信息......这真是太神奇了。 再一次，谢谢@seaborn！ 你让我像'贾格尔'一样移动！




```python
#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();
```

Although we already know some of the main figures, this mega scatter plot gives us a reasonable idea about variables relationships.

One of the figures we may find interesting is the one between 'TotalBsmtSF' and 'GrLiveArea'. In this figure we can see the dots drawing a linear line, which almost acts like a border. It totally makes sense that the majority of the dots stay below that line. Basement areas can be equal to the above ground living area, but it is not expected a basement area bigger than the above ground living area (unless you're trying to buy a bunker).

The plot concerning 'SalePrice' and 'YearBuilt' can also make us think. In the bottom of the 'dots cloud', we see what almost appears to be a shy exponential function (be creative). We can also see this same tendency in the upper limit of the 'dots cloud' (be even more creative). Also, notice how the set of dots regarding the last years tend to stay above this limit (I just wanted to say that prices are increasing faster now).

Ok, enough of Rorschach test for now. Let's move forward to what's missing: missing data!

虽然我们已经知道了一些主要数据，但这个巨大的散点图给出了关于变量关系的合理概念。

我们可能感兴趣的一个数字是'TotalBsmtSF'和'GrLiveArea'之间的数字。在这个图中，我们可以看到绘制一条直线的点，这几乎就像一个边界。大多数点都低于该线，这是完全有道理的。地下室区域可以与地上生活区域相等，但预计地下室区域不会超过地上生活区域（除非您尝试购买地堡）。

关于'SalePrice'和'YearBuilt'的情节也可以让我们思考。在“点云”的底部，我们看到几乎看起来像一个害羞的指数函数（具有创造性）。我们也可以在“点云”的上限看到同样的趋势（更具创造性）。此外，请注意关于过去几年的点集如何保持高于此限制（我只是想说价格现在增长得更快）。

好的，现在Rorschach测试已经足够了。让我们继续前进到缺少的东西：缺失值！

# 4. Missing data

Important questions when thinking about missing data:

* How prevalent is the missing data?
* Is missing data random or does it have a pattern?

The answer to these questions is important for practical reasons because missing data can imply a reduction of the sample size. This can prevent us from proceeding with the analysis. Moreover, from a substantive perspective, we need to ensure that the missing data process is not biased and hidding an inconvenient truth.

# 4.缺失值

在考虑缺少数据时的重要问题：

*缺失数据有多普遍？
*随机丢失数据还是有模式？

出于实际原因，这些问题的答案很重要，因为缺少数据可能意味着样本量的减少。 这可以阻止我们继续进行分析。 此外，从实质的角度来看，我们需要确保缺失的数据流程没有偏见并隐藏一个不方便的事实。


```python
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
```

Let's analyse this to understand how to handle the missing data.

We'll consider that when more than 15% of the data is missing, we should delete the corresponding variable and pretend it never existed. This means that we will not try any trick to fill the missing data in these cases. According to this, there is a set of variables (e.g. 'PoolQC', 'MiscFeature', 'Alley', etc.) that we should delete. The point is: will we miss this data? I don't think so. None of these variables seem to be very important, since most of them are not aspects in which we think about when buying a house (maybe that's the reason why data is missing?). Moreover, looking closer at the variables, we could say that variables like 'PoolQC', 'MiscFeature' and 'FireplaceQu' are strong candidates for outliers, so we'll be happy to delete them.

In what concerns the remaining cases, we can see that 'Garage*X*' variables have the same number of missing data. I bet missing data refers to the same set of observations (although I will not check it; it's just 5% and we should not spend 20$ in 5$ problems). Since the most important information regarding garages is expressed by 'GarageCars' and considering that we are just talking about 5% of missing data, I'll delete the mentioned 'Garage*X*' variables. The same logic applies to 'Bsmt*X*' variables.

Regarding 'MasVnrArea' and 'MasVnrType', we can consider that these variables are not essential. Furthermore, they have a strong correlation with 'YearBuilt' and 'OverallQual' which are already considered. Thus, we will not lose information if we delete 'MasVnrArea' and 'MasVnrType'.

Finally, we have one missing observation in 'Electrical'. Since it is just one observation, we'll delete this observation and keep the variable.

In summary, to handle missing data, we'll delete all the variables with missing data, except the variable 'Electrical'. In 'Electrical' we'll just delete the observation with missing data.

让我们分析一下，了解如何处理丢失的数据。

我们会考虑当超过15％的数据丢失时，我们应该删除相应的变量并假装它从未存在过。这意味着在这些情况下我们不会尝试任何技巧来填充缺失的数据。据此，我们应该删除一组变量（例如'PoolQC'，'MiscFeature'，'Alley'等）。关键是：我们会错过这些数据吗？我不这么认为。这些变量似乎都不是非常重要，因为它们中的大多数都不是我们在购买房屋时考虑的方面（也许这就是数据缺失的原因？）。此外，仔细观察变量，我们可以说像'PoolQC'，'MiscFeature'和'FireplaceQu'这样的变量是异常值的强大候选者，所以我们很乐意删除它们。

在剩下的情况下，我们可以看到'Garage * X *'变量具有相同数量的缺失数据。我打赌缺少的数据是指同一组观察结果（虽然我不会检查它;它只是5％而我们不应该在5美元的问题上花费20美元）。由于关于车库的最重要的信息是由'GarageCars'表示的，并且考虑到我们只讨论了5％的缺失数据，我将删除提到的'Garage * X *'变量。相同的逻辑适用于'Bsmt * X *'变量。

关于'MasVnrArea'和'MasVnrType'，我们可以认为这些变量不是必需的。此外，它们与已经考虑过的'YearBuilt'和'OverallQual'有很强的相关性。因此，如果我们删除'MasVnrArea'和'MasVnrType'，我们不会丢失信息。

最后，我们在'Electrical'中有一个缺失的观察结果。因为它只是一个观察，我们将删除这个观察并保留变量。

总之，为了处理丢失的数据，我们将删除所有缺少数据的变量，但变量“Electrical”除外。在“电气”中，我们将删除缺少数据的观察结果。


```python
#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() #just checking that there's no missing data missing...
```

# Out liars!

Outliers is also something that we should be aware of. Why? Because outliers can markedly affect our models and can be a valuable source of information, providing us insights about specific behaviours.

Outliers is a complex subject and it deserves more attention. Here, we'll just do a quick analysis through the standard deviation of 'SalePrice' and a set of scatter plots.

# 淘汰者！

异常值也是我们应该注意的事情。 为什么？ 因为异常值可以显着影响我们的模型，并且可以成为有价值的信息来源，为我们提供有关特定行为的见解。

异常值是一个复杂的主题，值得更多关注。 在这里，我们将通过'SalePrice'的标准偏差和一组散点图进行快速分析。

### Univariate analysis
### 单变量分析

The primary concern here is to establish a threshold that defines an observation as an outlier. To do so, we'll standardize the data. In this context, data standardization means converting data values to have mean of 0 and a standard deviation of 1.

这里主要关注的是建立一个阈值，将观察定义为异常值。 为此，我们将标准化数据。 在这种情况下，数据标准化意味着将数据值转换为平均值为0且标准偏差为1。


```python
#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
```

How 'SalePrice' looks with her new clothes:

* Low range values are similar and not too far from 0.
* High range values are far from 0 and the 7.something values are really out of range.

For now, we'll not consider any of these values as an outlier but we should be careful with those two 7.something values.

“SalePrice”如何看待她的新衣服：

*低范围值与0相似且不太远。
*高范围值远离0，而7.something值实际上超出范围。

目前，我们不会将这些值中的任何一个视为异常值，但我们应该小心这两个7.something值。

### Bivariate analysis

We already know the following scatter plots by heart. However, when we look to things from a new perspective, there's always something to discover. As Alan Kay said, 'a change in perspective is worth 80 IQ points'.


```python
#bivariate analysis saleprice/grlivarea
# 双变量分析
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
```

What has been revealed:

* The two values with bigger 'GrLivArea' seem strange and they are not following the crowd. We can speculate why this is happening. Maybe they refer to agricultural area and that could explain the low price. I'm not sure about this but I'm quite confident that these two points are not representative of the typical case. Therefore, we'll define them as outliers and delete them.
* The two observations in the top of the plot are those 7.something observations that we said we should be careful about. They look like two special cases, however they seem to be following the trend. For that reason, we will keep them.

揭晓的内容：

*具有更大“GrLivArea”的两个值似乎很奇怪，他们并没有跟随人群。 我们可以推测为什么会这样。 也许他们指的是农业区，这可以解释低价格。 我不确定这一点，但我完全相信这两点并不代表典型案例。 因此，我们将它们定义为异常值并删除它们。
*图中顶部的两个观察结果是我们应该注意的那些观察结果。 它们看起来像两个特例，但它们似乎跟随潮流。 出于这个原因，我们将保留它们。



```python
#deleting points
# 删除异常点
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
```


```python
#bivariate analysis saleprice/grlivarea
# 双变量分析
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
```

We can feel tempted to eliminate some observations (e.g. TotalBsmtSF > 3000) but I suppose it's not worth it. We can live with that, so we'll not do anything.

我们可以尝试消除一些观察结果（例如TotalBsmtSF> 3000），但我认为这不值得。 我们可以忍受，所以我们什么都不做。

# 5. Getting hard core
# 5.获得核心

In Ayn Rand's novel, 'Atlas Shrugged', there is an often-repeated question: who is John Galt? A big part of the book is about the quest to discover the answer to this question.

I feel Randian now. Who is 'SalePrice'?

The answer to this question lies in testing for the assumptions underlying the statistical bases for multivariate analysis. We already did some data cleaning and discovered a lot about 'SalePrice'. Now it's time to go deep and understand how 'SalePrice' complies with the statistical assumptions that enables us to apply multivariate techniques.

According to [Hair et al. (2013)](https://amzn.to/2uC3j9p), four assumptions should be tested:

* <b>Normality</b> - When we talk about normality what we mean is that the data should look like a normal distribution. This is important because several statistic tests rely  on this (e.g. t-statistics). In this exercise we'll just check univariate normality for 'SalePrice' (which is a limited approach). Remember that univariate normality doesn't ensure multivariate normality (which is what we would like to have), but it helps. Another detail to take into account is that in big samples (>200 observations) normality is not such an issue. However, if we solve normality, we avoid a lot of other problems (e.g. heteroscedacity) so that's the main reason why we are doing this analysis.

* <b>Homoscedasticity</b> - I just hope I wrote it right. Homoscedasticity refers to the 'assumption that dependent variable(s) exhibit equal levels of variance across the range of predictor variable(s)' [(Hair et al., 2013)](https://amzn.to/2uC3j9p). Homoscedasticity is desirable because we want the error term to be the same across all values of the independent variables.

* <b>Linearity</b>- The most common way to assess linearity is to examine scatter plots and search for linear patterns. If patterns are not linear, it would be worthwhile to explore data transformations. However, we'll not get into this because most of the scatter plots we've seen appear to have linear relationships.

* <b>Absence of correlated errors</b> - Correlated errors, like the definition suggests, happen when one error is correlated to another. For instance, if one positive error makes a negative error systematically, it means that there's a relationship between these variables. This occurs often in time series, where some patterns are time related. We'll also not get into this. However, if you detect something, try to add a variable that can explain the effect you're getting. That's the most common solution for correlated errors.

What do you think Elvis would say about this long explanation? 'A little less conversation, a little more action please'? Probably... By the way, do you know what was Elvis's last great hit?

(...)

The bathroom floor.

在艾恩兰德的小说“阿特拉斯耸耸肩”中，有一个经常重复的问题：谁是约翰高尔特？本书的一大部分内容是探索这个问题的答案。

我现在感觉到了Randian。谁是'SalePrice'？

这个问题的答案在于测试多变量分析统计基础的假设。我们已经做了一些数据清理并发现了很多关于'SalePrice'的信息。现在是时候深入了解'SalePrice'如何符合统计假设，使我们能够应用多变量技术。

根据[Hair et al。 （2013）]（https://amzn.to/2uC3j9p），应测试四个假设：

* <b>正常性</b>  - 当我们谈论正常性时，我们的意思是数据应该看起来像正态分布。这很重要，因为有几项统计测试依赖于此（例如t统计）。在本练习中，我们将只检查'SalePrice'的单变量正态性（这是一种有限的方法）。请记住，单变量正态性不能确保多元正态性（这是我们想要的），但它有所帮助。需要考虑的另一个细节是，在大样本（> 200个观测值）中，正常性不是这样的问题。但是，如果我们解决正态性，我们会避免很多其他问题（例如异方差），这就是我们进行此分析的主要原因。

* <b> Homoscedasticity </b>  - 我希望我写得对。同方差性是指“因变量（s）在预测变量范围内表现出相等的方差水平的假设”[（Hair et al。，2013）]（https://amzn.to/2uC3j9p）。同方差性是可取的，因为我们希望误差项在自变量的所有值上都是相同的。

* <b>线性</b>  - 评估线性度的最常用方法是检查散点图并搜索线性模式。如果模式不是线性的，那么探讨数据转换是值得的。但是，我们不会深入研究这一点，因为我们看到的大多数散点图似乎都有线性关系。

* <b>缺少相关错误</b>  - 当一个错误与另一个错误相关时，相关错误（如定义所示）就会发生。例如，如果一个正误差系统地产生负面误差，则意味着这些变量之间存在关系。这通常以时间序列发生，其中一些模式与时间相关。我们也不会涉及到这一点。但是，如果您检测到某些内容，请尝试添加一个可以解释您所获得效果的变量。这是相关错误的最常见解决方案。

你认为埃尔维斯对这个漫长的解释有什么看法？ “请少一点谈话，请多一点动作”？可能......顺便问一下，你知道猫王的最后一次重击是什么吗？

（......）

浴室地板。

### In the search for normality
### 在寻找常态

The point here is to test 'SalePrice' in a very lean way. We'll do this paying attention to:

* <b>Histogram</b> - Kurtosis and skewness.
* <b>Normal probability plot</b> - Data distribution should closely follow the diagonal that represents the normal distribution.

这里的重点是以非常精益的方式测试'SalePrice'。 我们这样做会注意：

* <b>直方图</b>  - 峰度和偏度。
* <b>正态概率图</b>  - 数据分布应紧密跟随代表正态分布的对角线。


```python
#histogram and normal probability plot
#直方图和正态概率图
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
```

Ok, 'SalePrice' is not normal. It shows 'peakedness', positive skewness and does not follow the diagonal line.

But everything's not lost. A simple data transformation can solve the problem. This is one of the awesome things you can learn in statistical books: in case of positive skewness, log transformations usually works well. When I discovered this, I felt like an Hogwarts' student discovering a new cool spell.

*Avada kedavra!*

好的，'SalePrice'不正常。 它显示“峰值”，正偏斜并且不遵循对角线。

但一切都没有丢失。 简单的数据转换可以解决问题。 这是你可以在统计书中学到的很棒的东西之一：如果是正偏斜，日志转换通常很有效。 当我发现这一点时，我感觉自己就像霍格沃茨的学生发现了一个新的酷咒。

* Avada kedavra！*


```python
#applying log transformation
#应该日志转换
df_train['SalePrice'] = np.log(df_train['SalePrice'])
```


```python
#transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
```

Done! Let's check what's going on with 'GrLivArea'.
完成！ 让我们来看看'GrLivArea'发生了什么。


```python
#histogram and normal probability plot
#直方图和正态概率图
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
```

Tastes like skewness... *Avada kedavra!*


```python
#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
```


```python
#transformed histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)
```

Next, please...


```python
#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
```

Ok, now we are dealing with the big boss. What do we have here?

* Something that, in general, presents skewness.
* A significant number of observations with value zero (houses without basement).
* A big problem because the value zero doesn't allow us to do log transformations.

To apply a log transformation here, we'll create a variable that can get the effect of having or not having basement (binary variable). Then, we'll do a log transformation to all the non-zero observations, ignoring those with value zero. This way we can transform data, without losing the effect of having or not basement.

I'm not sure if this approach is correct. It just seemed right to me. That's what I call 'high risk engineering'.

好的，现在我们正在与大老板打交道。 我们有什么在这里？

*通常会出现偏斜的东西。
*大量观测值为零（没有地下室的房屋）。
*一个很大的问题，因为零值不允许我们进行日志转换。

要在此处应用日志转换，我们将创建一个可以获得具有或不具有基础（二进制变量）的效果的变量。 然后，我们将对所有非零观测值进行对数转换，忽略值为零的观测值。 这样我们就可以转换数据，而不会失去有或没有地下室的影响。

我不确定这种方法是否正确。 这对我来说似乎是对的。 这就是我所谓的“高风险工程”。


```python
#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
#为新变量创建列（一个是足够的，因为它是二进制分类特征）
#如果area> 0则为1，对于area == 0则为0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
```


```python
#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
```


```python
#histogram and normal probability plot
#直方图和正态概率图
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
```

### In the search for writing 'homoscedasticity' right at the first attempt
### 在第一次尝试时寻找“方差齐性”的写作

The best approach to test homoscedasticity for two metric variables is graphically. Departures from an equal dispersion are shown by such shapes as cones (small dispersion at one side of the graph, large dispersion at the opposite side) or diamonds (a large number of points at the center of the distribution).

Starting by 'SalePrice' and 'GrLivArea'...

测试两个度量变量的同方差性的最佳方法是图形。 通过诸如锥体（在图的一侧的小色散，在相对侧的大的色散）或菱形（在分布的中心处的大量点）的形状示出了相等色散的偏离。

从'SalePrice'和'GrLivArea'开始......


```python
#scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);
```

Older versions of this scatter plot (previous to log transformations), had a conic shape (go back and check 'Scatter plots between 'SalePrice' and correlated variables (move like Jagger style)'). As you can see, the current scatter plot doesn't have a conic shape anymore. That's the power of normality! Just by ensuring normality in some variables, we solved the homoscedasticity problem.

Now let's check 'SalePrice' with 'TotalBsmtSF'.

此散点图的旧版本（在日志转换之前）具有圆锥形状（返回并检查'SalePrice'和相关变量之间的散点图（像Jagger样式一样移动）'）。 如您所见，当前的散点图不再具有圆锥形状。 这是正常的力量！ 只是通过确保某些变量的正态性，我们解决了同方差性问题。

现在让我们用'TotalBsmtSF'检查'SalePrice'。


```python
#scatter plot
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);
```

We can say that, in general, 'SalePrice' exhibit equal levels of variance across the range of 'TotalBsmtSF'. Cool!

我们可以说，一般来说，'SalePrice'在'TotalBsmtSF'范围内表现出相同的差异水平。 酷！

# Last but not the least, dummy variables

Easy mode.


```python
#convert categorical variable into dummy
df_train = pd.get_dummies(df_train)
```

# Conclusion

That's it! We reached the end of our exercise.

Throughout this kernel we put in practice many of the strategies proposed by [Hair et al. (2013)](https://amzn.to/2uC3j9p). We philosophied about the variables, we analysed 'SalePrice' alone and with the most correlated variables, we dealt with missing data and outliers, we tested some of the fundamental statistical assumptions and we even transformed categorial variables into dummy variables. That's a lot of work that Python helped us make easier.

But the quest is not over. Remember that our story stopped in the Facebook research. Now it's time to give a call to 'SalePrice' and invite her to dinner. Try to predict her behaviour. Do you think she's a girl that enjoys regularized linear regression approaches? Or do you think she prefers ensemble methods? Or maybe something else?

It's up to you to find out.

而已！ 我们的练习结束了。

在整个内核中，我们实践了[Hair等人提出的许多策略。（2013）]（https://amzn.to/2uC3j9p）。 我们对变量进行了哲学思考，我们单独分析了“SalePrice”，并且使用最相关的变量，我们处理了缺失数据和异常值，我们测试了一些基本的统计假设，我们甚至将分类变量转换为虚拟变量。 这是Python帮助我们轻松完成的大量工作。

但是这个任务还没有结束。 请记住，我们的故事在Facebook研究中停止了。 现在是时候打电话给'SalePrice'并邀请她吃饭。 试着预测她的行为。 你认为她是一个喜欢正则化线性回归方法的女孩吗？ 或者你认为她更喜欢合奏方法？ 或者别的什么？

这取决于你找出答案。

# <b>References</b>
* [My blog](http://pmarcelino.com)
* [My other kernels](https://www.kaggle.com/pmarcelino/data-analysis-and-feature-extraction-with-python)
* [Hair et al., 2013, Multivariate Data Analysis, 7th Edition](https://amzn.to/2JuDmvo)

# Acknowledgements

Thanks to [João Rico](https://www.linkedin.com/in/joaomiguelrico/) for reading drafts of this.
