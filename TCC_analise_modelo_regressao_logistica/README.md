
# MODELO 1 - Avaliação do Modelo de Classificação.
### Introdução.
##### Este  Notebook é destina a avaliação do modelo de regressão logística e separação dos dados  no arquivo voice_fix.csv


---
---
---
---
---


##  Resumo da análise anterior com a base tratada em python das propriedades acústicas.


```python
%matplotlib inline
```


```python
# Importa as bibliotecas
import pandas
import matplotlib.pyplot as plt
import numpy 
#from pandas.tools.plotting import scatter_matrix
from  pandas.plotting  import scatter_matrix
import seaborn as sb
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import Normalizer
#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score , roc_curve, auc ,accuracy_score,recall_score, precision_score
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix





```


```python
url = "C:\\Users\\jorge\\Desktop\\TCC\\tcc_to_git\\tcc\\baseDados\\voice_fix.csv"
colunas = ["meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode","centroid","meanfun","minfun","maxfun","meandom","mindom","maxdom","dfrange","modindx","label"]
dataset = pandas.read_csv(url, names=colunas , sep = ",")
```


```python
dataset[["meanfreq","sd","median"]].head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>meanfreq</th>
      <th>sd</th>
      <th>median</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.172557</td>
      <td>0.064241</td>
      <td>0.176893</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.172557</td>
      <td>0.067310</td>
      <td>0.176893</td>
    </tr>
  </tbody>
</table>
</div>



## 1) Refazendo boxplot.
O BOXPLOT representa os dados através de um retângulo
construído com os quartis e fornece informação sobre valores
extremos. 


```python
## Separação dos dados pela classe label, vozes de homens e mulheres.
dfHomens = dataset[dataset["label"] == "male"]
dfMulheres = dataset[dataset["label"] == "female"]
```

### Dataframe da classe femele


```python
plt.rcParams['figure.figsize'] = (20,15)
dfMulheres[colunas[0:20]].plot(kind='box', subplots=True, layout=(4,5), sharex=False, sharey=False,fontsize=18)
plt.show()
```


![png](output_10_0.png)


### Dataframe da classe male


```python
plt.rcParams['figure.figsize'] = (20,15)
dfHomens[colunas[0:20]].plot(kind='box', subplots=True, layout=(4,5), sharex=False, sharey=False,fontsize=18)
plt.show()
```


![png](output_12_0.png)


## Fim do resumo análise.


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score , roc_curve, auc

```


```python
url = ".\\baseDados\\voice_fix.csv"
colunas = ["meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode","centroid","meanfun","minfun","maxfun","meandom","mindom","maxdom","dfrange","modindx","label"]
dataset = pandas.read_csv(url, names=colunas , sep = ",")
```

---


---



---
# Procedimentos de avaliação de modelo
Train/Test Split
K-Fold Cross Validation

##  2)  Preparando a base para usar no modelo de regressão logística.


```python
print(dataset.head(2).transpose())
```

                      0           1
    meanfreq   0.172557    0.172557
    sd        0.0642413     0.06731
    median     0.176893    0.176893
    Q25        0.121089    0.121089
    Q75        0.227842    0.227842
    IQR        0.109055    0.109055
    skew        1.90605     1.90605
    kurt        6.45022     6.45022
    sp.ent     0.893369    0.892193
    sfm        0.491918    0.513724
    mode              0           0
    centroid   0.172557    0.172557
    meanfun   0.0842791    0.107937
    minfun    0.0157017   0.0158259
    maxfun     0.275862    0.273863
    meandom   0.0078125  0.00901442
    mindom    0.0078125   0.0078125
    maxdom    0.0078125   0.0546875
    dfrange           0    0.046875
    modindx    0.132999    0.124688
    label          male        male
    

##  3)  Atribuindo para female=1 (Mulheres), male=0 (Homens) e adicionando a coluna gênero para representar a classe como dummy.


```python
df_pre = dataset

df_pre['genero'] = df_pre['label'].replace({'female': 1, 'male': 0})
dataset = df_pre
```


```python
print(df_pre.head(2).transpose())



#dataset = df_pre
```

                      0           1
    meanfreq   0.172557    0.172557
    sd        0.0642413     0.06731
    median     0.176893    0.176893
    Q25        0.121089    0.121089
    Q75        0.227842    0.227842
    IQR        0.109055    0.109055
    skew        1.90605     1.90605
    kurt        6.45022     6.45022
    sp.ent     0.893369    0.892193
    sfm        0.491918    0.513724
    mode              0           0
    centroid   0.172557    0.172557
    meanfun   0.0842791    0.107937
    minfun    0.0157017   0.0158259
    maxfun     0.275862    0.273863
    meandom   0.0078125  0.00901442
    mindom    0.0078125   0.0078125
    maxdom    0.0078125   0.0546875
    dfrange           0    0.046875
    modindx    0.132999    0.124688
    label          male        male
    genero            0           0
    


```python
#df =dataset.rename(columns={'label': 'genero'})
print(df_pre.tail(2).transpose())
```

                   3166       3167
    meanfreq   0.143659   0.165509
    sd        0.0906283  0.0928835
    median     0.184976   0.183044
    Q25        0.181927   0.181927
    Q75        0.219943   0.250827
    IQR       0.0412693  0.0412693
    skew        1.59106    1.70503
    kurt         5.3883    5.76912
    sp.ent     0.950436   0.938829
    sfm         0.67547   0.601529
    mode       0.212202   0.201041
    centroid   0.143659   0.165509
    meanfun    0.172375   0.185607
    minfun    0.0344828  0.0622568
    maxfun     0.274763   0.271186
    meandom     0.79136   0.227022
    mindom    0.0078125  0.0078125
    maxdom      3.59375   0.554688
    dfrange     3.58594   0.546875
    modindx    0.133931   0.133931
    label        female     female
    genero            1          1
    

#  4)   Dataset: Train/Test Split para os modelos.
Esse método divide o conjunto de dados em duas partes: um conjunto de treinamento e um conjunto de testes. O conjunto de treinamento é usado para treinar o modelo. Também podemos medir a precisão do modelo no conjunto de treinamento.

Logistic Regression coefficients na formula:
 y=  1 * b0 + b1*X1 + b2*X2+ b3*Xn

 ##   5)  Criando explicitamente  y-intercept: b0. 


```python
df_pre['int']=1
print(df_pre.head().transpose())
```

                      0           1           2          3          4
    meanfreq   0.172557    0.172557    0.172557   0.151228    0.13512
    sd        0.0642413     0.06731   0.0635487  0.0612157  0.0627691
    median     0.176893    0.176893    0.176893   0.158011   0.124656
    Q25        0.121089    0.121089    0.121089  0.0965817  0.0787202
    Q75        0.227842    0.227842    0.227842   0.207955   0.206045
    IQR        0.109055    0.109055    0.123207   0.111374   0.127325
    skew        1.90605     1.90605     1.90605    1.23283    1.10117
    kurt        6.45022     6.45022     6.45022     4.1773    4.33371
    sp.ent     0.893369    0.892193    0.918553   0.963322   0.971955
    sfm        0.491918    0.513724    0.478905   0.727232   0.783568
    mode              0           0           0  0.0838782   0.104261
    centroid   0.172557    0.172557    0.172557   0.151228    0.13512
    meanfun   0.0842791    0.107937   0.0987063  0.0889648   0.106398
    minfun    0.0157017   0.0158259   0.0156556  0.0177976  0.0169312
    maxfun     0.275862    0.273863    0.271186   0.273863   0.275166
    meandom   0.0078125  0.00901442  0.00799006   0.201497   0.712812
    mindom    0.0078125   0.0078125   0.0078125  0.0078125  0.0078125
    maxdom    0.0078125   0.0546875    0.015625     0.5625    5.48438
    dfrange           0    0.046875   0.0078125   0.554688    5.47656
    modindx    0.132999    0.124688    0.124688   0.130223   0.124688
    label          male        male        male       male       male
    genero            0           0           0          0          0
    int               1           1           1          1          1
    


```python
## Separação dos dados pela classe label, vozes de homens e mulheres.
df_male = df_pre[df_pre["label"] == "male"]
df_female = df_pre[df_pre["label"] == "female"]




```


```python
print(df_male.head().transpose())
```

                      0           1           2          3          4
    meanfreq   0.172557    0.172557    0.172557   0.151228    0.13512
    sd        0.0642413     0.06731   0.0635487  0.0612157  0.0627691
    median     0.176893    0.176893    0.176893   0.158011   0.124656
    Q25        0.121089    0.121089    0.121089  0.0965817  0.0787202
    Q75        0.227842    0.227842    0.227842   0.207955   0.206045
    IQR        0.109055    0.109055    0.123207   0.111374   0.127325
    skew        1.90605     1.90605     1.90605    1.23283    1.10117
    kurt        6.45022     6.45022     6.45022     4.1773    4.33371
    sp.ent     0.893369    0.892193    0.918553   0.963322   0.971955
    sfm        0.491918    0.513724    0.478905   0.727232   0.783568
    mode              0           0           0  0.0838782   0.104261
    centroid   0.172557    0.172557    0.172557   0.151228    0.13512
    meanfun   0.0842791    0.107937   0.0987063  0.0889648   0.106398
    minfun    0.0157017   0.0158259   0.0156556  0.0177976  0.0169312
    maxfun     0.275862    0.273863    0.271186   0.273863   0.275166
    meandom   0.0078125  0.00901442  0.00799006   0.201497   0.712812
    mindom    0.0078125   0.0078125   0.0078125  0.0078125  0.0078125
    maxdom    0.0078125   0.0546875    0.015625     0.5625    5.48438
    dfrange           0    0.046875   0.0078125   0.554688    5.47656
    modindx    0.132999    0.124688    0.124688   0.130223   0.124688
    label          male        male        male       male       male
    genero            0           0           0          0          0
    int               1           1           1          1          1
    


```python
print(df_female.head().transpose())
```

                   1584       1585       1586       1587       1588
    meanfreq   0.158108   0.182855   0.199807    0.19528   0.208504
    sd        0.0827816  0.0677889  0.0619738  0.0720869  0.0575502
    median     0.191191   0.200639   0.211358   0.204656   0.220229
    Q25        0.181927   0.175489   0.184422   0.180611   0.190343
    Q75        0.224552   0.226068   0.235687   0.255954   0.249759
    IQR       0.0412693  0.0505788  0.0512645  0.0403311  0.0594155
    skew        2.80134    3.00189    2.54384    2.39233    1.70779
    kurt        9.34563    9.34563     14.922    10.0615    5.67091
    sp.ent     0.952161   0.910458   0.904432   0.907115   0.879674
    sfm        0.679223   0.506099   0.425289   0.524209   0.343548
    mode       0.201834   0.201834   0.201834   0.193435   0.201834
    centroid   0.158108   0.182855   0.199807    0.19528   0.208504
    meanfun    0.185042    0.15959   0.156465   0.182629   0.162043
    minfun    0.0230216  0.0187135  0.0161943  0.0249221  0.0168067
    maxfun     0.275862   0.275927   0.275927   0.275862   0.275927
    meandom    0.272964    0.25897   0.250446   0.269531   0.260789
    mindom     0.046875  0.0546875  0.0546875  0.0546875  0.0546875
    maxdom     0.742188   0.804688   0.898438   0.703125     0.8125
    dfrange    0.695312       0.75    0.84375   0.648438   0.757812
    modindx    0.133931   0.129735   0.133931   0.133931   0.129735
    label        female     female     female     female     female
    genero            1          1          1          1          1
    int               1          1          1          1          1
    

### Separando X e Y para dataframe_female


```python
X_entrada_female = df_female.drop(columns=['label','genero'])
Y_entrada_female = df_female['genero']

```


```python
print(X_entrada_female.head().transpose())

feature_cols=X_entrada_female.columns
feature_cols

```

                  1584      1585       1586       1587      1588
    meanfreq  0.158108  0.182855   0.199807   0.195280  0.208504
    sd        0.082782  0.067789   0.061974   0.072087  0.057550
    median    0.191191  0.200639   0.211358   0.204656  0.220229
    Q25       0.181927  0.175489   0.184422   0.180611  0.190343
    Q75       0.224552  0.226068   0.235687   0.255954  0.249759
    IQR       0.041269  0.050579   0.051265   0.040331  0.059416
    skew      2.801344  3.001890   2.543841   2.392326  1.707786
    kurt      9.345630  9.345630  14.921964  10.061489  5.670912
    sp.ent    0.952161  0.910458   0.904432   0.907115  0.879674
    sfm       0.679223  0.506099   0.425289   0.524209  0.343548
    mode      0.201834  0.201834   0.201834   0.193435  0.201834
    centroid  0.158108  0.182855   0.199807   0.195280  0.208504
    meanfun   0.185042  0.159590   0.156465   0.182629  0.162043
    minfun    0.023022  0.018713   0.016194   0.024922  0.016807
    maxfun    0.275862  0.275927   0.275927   0.275862  0.275927
    meandom   0.272964  0.258970   0.250446   0.269531  0.260789
    mindom    0.046875  0.054688   0.054688   0.054688  0.054688
    maxdom    0.742188  0.804688   0.898438   0.703125  0.812500
    dfrange   0.695312  0.750000   0.843750   0.648438  0.757812
    modindx   0.133931  0.129735   0.133931   0.133931  0.129735
    int       1.000000  1.000000   1.000000   1.000000  1.000000
    




    Index(['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt',
           'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 'maxfun',
           'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx', 'int'],
          dtype='object')




```python
print(Y_entrada_female.head())
```

    1584    1
    1585    1
    1586    1
    1587    1
    1588    1
    Name: genero, dtype: int64
    

### Separando X e Y para dataframe_male


```python
X_entrada_male = df_male.drop(columns=['label','genero'])
Y_entrada_male = df_male['genero']
```


```python
print(X_entrada_male.head().transpose())
```

                     0         1         2         3         4
    meanfreq  0.172557  0.172557  0.172557  0.151228  0.135120
    sd        0.064241  0.067310  0.063549  0.061216  0.062769
    median    0.176893  0.176893  0.176893  0.158011  0.124656
    Q25       0.121089  0.121089  0.121089  0.096582  0.078720
    Q75       0.227842  0.227842  0.227842  0.207955  0.206045
    IQR       0.109055  0.109055  0.123207  0.111374  0.127325
    skew      1.906048  1.906048  1.906048  1.232831  1.101174
    kurt      6.450221  6.450221  6.450221  4.177296  4.333713
    sp.ent    0.893369  0.892193  0.918553  0.963322  0.971955
    sfm       0.491918  0.513724  0.478905  0.727232  0.783568
    mode      0.000000  0.000000  0.000000  0.083878  0.104261
    centroid  0.172557  0.172557  0.172557  0.151228  0.135120
    meanfun   0.084279  0.107937  0.098706  0.088965  0.106398
    minfun    0.015702  0.015826  0.015656  0.017798  0.016931
    maxfun    0.275862  0.273863  0.271186  0.273863  0.275166
    meandom   0.007812  0.009014  0.007990  0.201497  0.712812
    mindom    0.007812  0.007812  0.007812  0.007812  0.007812
    maxdom    0.007812  0.054688  0.015625  0.562500  5.484375
    dfrange   0.000000  0.046875  0.007812  0.554688  5.476562
    modindx   0.132999  0.124688  0.124688  0.130223  0.124688
    int       1.000000  1.000000  1.000000  1.000000  1.000000
    


```python
print(Y_entrada_male.head())
```

    0    0
    1    0
    2    0
    3    0
    4    0
    Name: genero, dtype: int64
    

##  6)  Divisão balanceada de 30% teste e 70%  para o treino.

### Feito a divisão  randômica de 30 test e 70 treino no dataframe_female


```python
X_trainF,X_testF,y_trainF,y_testF = train_test_split(X_entrada_female,Y_entrada_female,test_size=0.30,random_state=0)
```

### Feito a divisão randômica de 30 test e 70 treino no dataframe_male


```python
X_trainM, X_testM, y_trainM ,y_testM = train_test_split(X_entrada_male,Y_entrada_male,test_size=0.30,random_state=0)
```

### Concatenando os datraframes  Após ad divisão dos dados de treino e test  male e frame


```python
X_train_frames = [X_trainF, X_trainM]
```


```python
X_test_frames = [X_testF,X_testM]
```


```python
y_test_frames = [y_testF, y_testM]
```


```python
y_train_frames = [ y_trainF,  y_trainM]
```

### Convertendo os datraframes  após a divisão dos dados de: treino e test,  male e frame


```python
X_train = pandas.concat(X_train_frames)
```


```python
X_test = pandas.concat(X_test_frames)
```


```python
y_train = pandas.concat(y_train_frames)
```


```python
y_test = pandas.concat(y_test_frames )
```

### Mostratandos as dimensões dos dados


```python
X_train.shape,X_test.shape , y_train.shape, y_test.shape


dictabela = {}
dictabela['Registros para treino'] = X_train.shape[0]
dictabela['Registros para teste'] = X_test.shape[0]



```


```python
dftreinoteste = pandas.DataFrame.from_dict(dictabela, orient="index").reset_index()
```


```python
dftreinoteste =dftreinoteste.rename(columns={'index': 'divisão dos dados'})
dftreinoteste =dftreinoteste.rename(columns={0: 'total'})
dftreinoteste

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>divisão dos dados</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Registros para treino</td>
      <td>2216</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Registros para teste</td>
      <td>952</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_train
```




    2858    1
    2040    1
    2394    1
    3133    1
    3005    1
           ..
    763     0
    835     0
    1216    0
    559     0
    684     0
    Name: genero, Length: 2216, dtype: int64



### Total de  voz  por classe, masculinas e femininas na base de treino


```python
dfContador =pandas.DataFrame(list(y_train), columns = ['genero'])
contagem = dfContador.groupby('genero').size()
print(contagem)

```

    genero
    0    1108
    1    1108
    dtype: int64
    

### Total de  voz  por classe, masculinas e femininas na base de teste


```python
dfContador =pandas.DataFrame(list(y_test), columns = ['genero'])
contagem = dfContador.groupby('genero').size()
print(contagem)
```

    genero
    0    476
    1    476
    dtype: int64
    

---
---
---

##  7)  Normalização dos dados por questão de escala.


```python
# Instantiate 
norm = Normalizer()

# Fit
norm.fit(X_train)

# Transform both training and testing sets
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)
```


```python
X_train_norm.shape , X_test_norm.shape
```




    ((2216, 21), (952, 21))




```python
print(X_train_norm)
```

    [[0.01070896 0.0013571  0.01063611 ... 0.4404305  0.00528321 0.04733426]
     [0.01080389 0.002876   0.01080535 ... 0.38455026 0.00730888 0.05951927]
     [0.01542367 0.00236176 0.01535375 ... 0.423673   0.00793265 0.07318508]
     ...
     [0.01959029 0.00592508 0.02281332 ... 0.490609   0.00824204 0.09827536]
     [0.01287192 0.00626938 0.011102   ... 0.49688596 0.01353424 0.10176224]
     [0.02327679 0.00906603 0.02096434 ... 0.10766927 0.02109371 0.15207357]]
    


#  8)  Salvando os dados de treino e teste em um dicionário serializado.


```python
dic_base_treino_test = {}
```


```python
dic_base_treino_test['y_train'] = y_train
```


```python
dic_base_treino_test['y_test'] = y_test
```


```python
dic_base_treino_test['X_train_norm'] = X_train_norm
```


```python
dic_base_treino_test['X_test_norm'] = X_test_norm
dic_base_treino_test['feature_cols'] =  feature_cols
```

### Salvando os dados para avaliação dos modelos


```python
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
```


```python
output = ".\\baseDados\\voice_treino_test.pk"
with open(output, 'wb') as pickle_file:
    pickle.dump(dic_base_treino_test, pickle_file)

```

---
---
---
---
---

#  9) Carregando os dados para avaliação do modelo


```python
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
```


```python
dic_base_treino_file = pickle.load(open( output, "rb" ))
```


```python
#print(dic_base_treino_file)
```


```python
y_train = dic_base_treino_file['y_train'] 
```


```python
 y_test = dic_base_treino_file['y_test'] 
```


```python
 X_train = dic_base_treino_file['X_train_norm']
```


```python
X_test = dic_base_treino_file['X_test_norm']
```


```python
dfContador =pandas.DataFrame(list(y_train), columns = ['genero'])
contagem = dfContador.groupby('genero').size()
print(contagem)
```

    genero
    0    1108
    1    1108
    dtype: int64
    


```python
dfContador =pandas.DataFrame(list(y_test), columns = ['genero'])
contagem = dfContador.groupby('genero').size()
print(contagem)
```

    genero
    0    476
    1    476
    dtype: int64
    

---


#  10)  Declarando o modelo.


```python
#logistic Regression
classifier = LogisticRegression(C=1, multi_class='ovr', penalty='l2', solver='liblinear')
```

#  11) Treinamento e teste do modelo.


```python
classifier.fit(X_train,y_train)
```




    LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='ovr', n_jobs=None, penalty='l2',
                       random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)




```python
y_pred=classifier.predict(X_test)
```

---

# 12) Modelo de avaliação de métricas.

##  16)  Classificação

###  Matriz de confusão.
Uma matriz de confusão pode ser definida livremente como uma tabela que descreve o desempenho de um modelo de classificação em um conjunto de dados de teste para os quais os valores verdadeiros são conhecidos.


```python
cm=confusion_matrix(y_test,y_pred)

```


```python
confusion_matrix_lda = pandas.DataFrame(cm, index = ['Negativos','Positivos'], columns = ['Previsão dos negativos','Previsão dos positivos'] )
confusion_matrix_lda['Total'] = 1
confusion_matrix_lda['Total'][0] = cm[0][0] + cm[0][1]
confusion_matrix_lda['Total'][1] = cm[1][0] + cm[1][1]
```


```python
confusion_matrix_lda
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Previsão dos negativos</th>
      <th>Previsão dos positivos</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Negativos</td>
      <td>406</td>
      <td>70</td>
      <td>476</td>
    </tr>
    <tr>
      <td>Positivos</td>
      <td>91</td>
      <td>385</td>
      <td>476</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(confusion_matrix_lda)
```

               Previsão dos negativos  Previsão dos positivos  Total
    Negativos                     406                      70    476
    Positivos                      91                     385    476
    


```python
#Plot the confusion matrix
plt.rcParams['figure.figsize'] = (10,5)
sb.set(font_scale=1.5)
sb.heatmap(cm, annot=True, fmt='g')
plt.show()
```


![png](output_98_0.png)


---

### True Positives:TP
Este valor indica a quantidade de registros que foram classificados como positivos corretamente.


```python
TP = confusion_matrix_lda['Previsão dos positivos'][1]
dfTP = pandas.DataFrame(TP, index = ['Positivos verdadeiros'], columns = ['Quantidade acertos'] )
```


```python
dfTP
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantidade acertos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Positivos verdadeiros</td>
      <td>385</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(dfTP)
```

                           Quantidade acertos
    Positivos verdadeiros                 385
    

---

### True Negatives:TN
Este valor indica a quantidade de registros que foram classificados como negativos de maneira correta.


```python
TN = confusion_matrix_lda['Previsão dos negativos'][0]
dfTN = pandas.DataFrame(TN, index = ['Verdadeiro Negativo'], columns = ['Quantidade acertos'] )
```


```python
dfTN
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantidade acertos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Verdadeiro Negativo</td>
      <td>406</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(dfTN)
```

                         Quantidade acertos
    Verdadeiro Negativo                 406
    

---

### Falso Positivos - False Positives:FP
Este valor indica a quantidade de registros que foram classificados como comentários positivos de maneira incorreta.


```python
FP = confusion_matrix_lda['Previsão dos positivos'][0]
dfFP = pandas.DataFrame(FP, index = ['Falso Positivo'], columns = ['Quantidade acertos'] )
```


```python
dfFP
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantidade acertos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Falso Positivo</td>
      <td>70</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(dfFP)
```

                    Quantidade acertos
    Falso Positivo                  70
    

---

### False Negatives:FN
Este valor indica a quantidade de registros que foram classificados como comentários negativos de maneira incorreta.


```python
FN = confusion_matrix_lda['Previsão dos negativos'][1]
dfFN = pandas.DataFrame(FN, index = ['Negativos Falsos'], columns = ['Quantidade acertos'] )
```


```python
dfFN
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantidade acertos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Negativos Falsos</td>
      <td>91</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(dfFN)
```

                      Quantidade acertos
    Negativos Falsos                  91
    

---

### Especificidade (Specificity)
Especificidade é a proporção de previsões negativas corretas para o total não de previsões negativas. Isso determina o grau de especificidade do classificador na previsão de instâncias positivas.

Specificity = (Numero de previsões negativas correta) / (Total do Numero Negativas prevista)

TN = / TN + FP


```python
Specificity = TN / float(TN + FP)
dfSpecificity = pandas.DataFrame(Specificity, index = ['Specificity'], columns = ['resultado'] )
```


```python
dfSpecificity
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>resultado</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Specificity</td>
      <td>0.852941</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(dfSpecificity)
```

                 resultado
    Specificity   0.852941
    

---

### Precisão Geral (Accuracy)
A precisão da classificação é a proporção de previsões corretas para o total não  de previsões. 

Accuracy = (numero de predições corretas / numero de predições)

$$ Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$


```python
#trés maneiras de fazer o caluclo
print((TP + TN) / float(TP + TN + FP + FN))
print(accuracy_score(y_test, y_pred))
print("Accuracy ", classifier.score(X_test, y_test)*100)
Accuracy= classifier.score(X_test, y_test)
```

    0.8308823529411765
    0.8308823529411765
    Accuracy  83.08823529411765
    


```python
dfAccuracy = pandas.DataFrame(Accuracy, index = ['Accuracy'], columns = ['resultado'] )
dfAccuracy
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>resultado</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Accuracy</td>
      <td>0.830882</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(dfAccuracy)
```

              resultado
    Accuracy   0.830882
    

---

### Sensibilidade ou recordação Recall




Sensibilidade ou recordação é a razão de previsões positivas corretas para o total não de previsões positivas, ou, mais simplesmente, quão sensível o classificador é para detectar instâncias positivas. Isso também é chamado de True Positive Rate

Recall = (Numero de positivas previstas corretamente) /( total de Predições positivas)

$$Recall = \frac{TP}{TP +FN}$$



```python
print(TP / float(TP + FN))
print(recall_score(y_test, y_pred))
Recall= recall_score(y_test, y_pred)
```

    0.8088235294117647
    0.8088235294117647
    


```python
dfRecall = pandas.DataFrame(Recall, index = ['Sensibilidade-Recall'], columns = ['resultado'] )
dfRecall
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>resultado</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Sensibilidade-Recall</td>
      <td>0.808824</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(dfRecall)
```

                          resultado
    Sensibilidade-Recall   0.808824
    

---

## Taxa positiva falsa (False Positive Rate)
A *false positive rate*, é a proporção de previsões negativas que foram determinadas como positivas para o número total de previsões negativas ou quando o valor real é negativo, com que frequência a previsão é incorreta.

FalsePositveRate = Números de falsos positivos / Total de predições negativas


$$FalsePositveRate = \frac{FP}{ TN + FP}$$



```python
print(FP / float(TN + FP))
FalsePositveRate = FP / float(TN + FP)
```

    0.14705882352941177
    


```python
dfFalsePositveRate = pandas.DataFrame(FalsePositveRate, index = ['Taxa de Falso Positvo'], columns = ['resultado'] )
dfFalsePositveRate
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>resultado</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Taxa de Falso Positvo</td>
      <td>0.147059</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(dfFalsePositveRate)
```

                           resultado
    Taxa de Falso Positvo   0.147059
    

---
### Precisão (Precision)
A precisão é a proporção de previsões corretas para o total  de não previsões preditas corretas. Isso mede a precisão do classificador ao prever instâncias positivas.

Precision = Número de positivas verdadeiras / Numero total de predicados positivos

$$Precision = \frac{TP} {TP + FP}$$


```python
print(TP / float(TP + FP))
print(precision_score(y_test, y_pred))
Precision = precision_score(y_test, y_pred)
```

    0.8461538461538461
    0.8461538461538461
    


```python
dfPrecision = pandas.DataFrame(Precision, index = ['Precisão'], columns = ['resultado'] )
dfPrecision
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>resultado</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Precisão</td>
      <td>0.846154</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(dfPrecision)
```

              resultado
    Precisão   0.846154
    

---

### F1 Score
O F1 Score é uma média harmônica entre precisão (que, apesar de ter o mesmo nome, não é a mesma citada acima) e recall. Veja abaixo as definições destes dois termos.

Ela é muito boa quando você possui um dataset com classes desproporcionais, e o seu modelo não emite probabilidades. Em geral, quanto maior o F1 score, melhor.



$$F1Score = \frac{2 \times Precisão \times Recall }{Precisão + Recall}$$



```python
F1Score = 2 * Precision *  Recall /  Precision + Recall
```


```python
print(F1Score)

```

    2.4264705882352944
    


```python
dfF1Score = pandas.DataFrame(F1Score, index = ['F1 Score'], columns = ['resultado'] )
dfF1Score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>resultado</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>F1 Score</td>
      <td>2.426471</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(dfF1Score)
```

              resultado
    F1 Score   2.426471
    

---
### 13) Curva ROC
Uma curva ROC é uma forma comumente usada para visualizar o desempenho de um classificador binário, significando um classificador com duas classes de saída possíveis. A curva plota a Taxa Positiva Real (Recall) contra a Taxa Falsa Positiva (também interpretada como Especificidade 1).


```python
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('Taxa de falsos positivos')
    plt.ylabel('Taxa de verdadeiros positivos')
    plt.title('Curva ROC:Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
```

#### Calcula a propabildade de previsão.


```python
y_pred_prob = classifier.predict_proba(X_test)[:, 1]
```


```python
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
```


```python
plot_roc_curve(fpr, tpr)
```


![png](output_145_0.png)


---

### AUC (área sob a curva) da Curva ROC
AUC ou Area Under the Curve é a porcentagem do gráfico do ROC que está abaixo da curva. AUC é útil como um único número de resumo do desempenho do classificador.


```python
print(roc_auc_score(y_test, y_pred_prob))
Auc=roc_auc_score(y_test, y_pred_prob)
```

    0.873181625591413
    


```python
dfAuc = pandas.DataFrame(Auc, index = ['AUC'], columns = ['resultado'] )
dfAuc
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>resultado</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AUC</td>
      <td>0.873182</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(dfAuc)
```

         resultado
    AUC   0.873182
    

# Salva dados para usar no gráfico consolidado.


```python
dic_logist={}
```


```python
dic_logist['Accuracy']=Accuracy
dic_logist['Auc']=Auc
dic_logist['y_pred_prob']=y_pred_prob
dic_logist['y_test']=y_test



```


```python
#dic_logist
```


```python
import pickle
```


```python
filename = '.\\baseDados\\regressaologitica.jss'
outfile = open(filename,'wb')
pickle.dump(dic_logist,outfile)
outfile.close()
```


```python
infile = open(filename,'rb')
test_dict = pickle.load(infile)
infile.close()
```


```python
#print(test_dict)

```


```python
print(type(test_dict))
```

    <class 'dict'>
    

# Fim de avaliação individual do modelo regressão logística

---
---
---
---
---


```python

```
