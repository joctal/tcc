# Análise exploratória.

### Introdução.

Este Jupyter Notebook investiga a base de dados de  propriedades acústicas disponíveis no site  http://www.primaryobjects.com/2016/06/22/identifying-the-gender-of-a-voice-using-machine-learning/   
Objetivo da investigação é determinar as chances de algum algoritmo para detecção de gênero, seja por estatística tradicional ou por meio técnicas machine learning e redes neurais, possibilitando a implantação em dispositivos embarcados de baixo custo de memória e processamento restrito. 

# Propriedades acústicas medidas

As seguintes propriedades acústicas de cada voz são medidas:

- **meanfreq**  : frequência média (em kHz) sobre as amostras compostas no sinal de arquivo de voz;
- **sd**  : desvio padrão da frequência, sobre as amostras compostas no sinal de arquivo de voz;
- **mediana**  : frequência mediana (em kHz) sobre as amostras compostas no sinal de arquivo de voz;
- **Q25**  : primeiro quantil (em kHz) sobre as amostras compostas no sinal de arquivo de voz;
- **Q75**  : terceiro quantil (em kHz) sobre as amostras compostas no sinal de arquivo de voz;
- **IQR**  : intervalo interquartil (em kHz)sobre as amostras compostas no sinal de arquivo de voz;
- **skew**  : média de assimetria da distribuição das frequências de vocal perdominante;
- **kurt**  : curtose distribuição espectral da voz, domínio da frequência;
- **sp.ent**  : entropia espectral, pureza da distribuição da voz em relação ao nível de ruído; 
- **sfm**  : nivelamento espectral,  estima a planaridade de um espectro de frequência;
- **modo**  : frequência de modo, ou seja, frequência dominante da voz;
- **centrod**  : frequência central máxima visto no domínio da frequência;
- **meanfun**  : média da frequência fundamental medida através do sinal acústico (Tonalidade base da voz);
- **minfun**  : frequência fundamental mínima medida no sinal acústico  (Tonalidade base da voz);
- **maxfun**  : frequência fundamental máxima medida através do sinal acústico (Tonalidade base da voz);
- **meandom**  : média da frequência dominante medida através do sinal acústico  (média total das notas  musicais mais graves da voz em relação ao sinal gravado);
- **mindom**  : mínimo de frequência dominante medido através do sinal acústico;
- **maxdom**  : máxima da frequência dominante medida através do sinal acústico;
- **dfrange**  : faixa de frequência dominante medida através do sinal acústico;
- **modindx**  : índice de modulação. Calculado como a diferença absoluta acumulada entre medições adjacentes de frequências fundamentais divididas pela faixa de frequência.
- **label**  : rotulo de identificador da amostra em relação ao sexo, adicionado durante a gravação "male" ou "female".

# Análise exploratória.

### TCC_ANALISE_Descritiva.ipynb (arquivo jupyter)
###  TCC_ANALISE_Descritiva.pdf   (resultados em pdf)
###  TCC_ANALISE_Descritiva.html   (resultados em html com as imagens)
###  R_TCC_ANALISE_EXPLORATORIA.Rmd  (análise feita em Rstudio)
###  R_TCC_ANALISE_EXPLORATORIA.pdf (resultados sem tratamento em R)

---

# Análise  regressão logística.

### TCC_analise_modelo_regressao_logistica.ipynb (arquivo jupyter)
### TCC_analise_modelo_regressao_logistica.pdf   (resultados em pdf)
### TCC_analise_modelo_regressao_logistica.html   (resultados em html com as imagens)





# Análise  Árvore de Classificação e Regressão ( CART ).

### TCC_CART.ipynb (arquivo jupyter)

### TCC_CART.pdf   (resultados em pdf)

### TCC_CART.html   (resultados em html com as imagens)




---
title: "TCC R  Notebook"
output:
  pdf_document: default
  html_notebook: default
---
# Analise exploratória.
### Introdução.
Este Jupyter Notebook investiga a base de dados de  propriedades acústicas disponível no site  http://www.primaryobjects.com/2016/06/22/identifying-the-gender-of-a-voice-using-machine-learning/   
Objetivo da investigação é determinar as chances de algum algoritmo para detecção de gênero, seja por estatística tradicional ou por meio técnicas machine learning e redes neurais, possibilitando a implantação em dispositivos embarcados de baixo custo de memoria e processamento restrito, para utilização de mídias inteligentes e interativas em lojas em moda.  




```{r}
#install.packages('Amelia')
#install.packages('corrplot')
#install.packages('caret')
#install.packages('ggplot2')
```

Carrega pacote com os dados usados no teste.

```{r}
library(mlbench)
library(e1071)
library(lattice)
library(Amelia)
library(corrplot)
library(caret)
datasetvoice = read.csv("C:\\Users\\jorge\\Desktop\\TCC\\00-PRATICA\\04-Resultados TCC\\baseDados\\Tvoice.csv",sep=',',header=T)
#==================================================
# Mostrar dados
#==================================================
#View(head(datasetvoice))
#View(tail(datasetvoice))
#print(head(datasetvoice))
```

Verificando alguns dados.


```{r}
head(datasetvoice, n=10)
```

Verifica a dimensão dos dados (linhas, colunas)

```{r}
dim(datasetvoice)
```

Verifica os tipos de dados de cada atributo método 1.

```{r}
sapply(datasetvoice, class)
```

Verifica os tipos de dados de cada atributo método 2.

```{r}
str(datasetvoice)
```

Estatística descritiva.

```{r}
summary(datasetvoice)
```
Distribuição das classes.

```{r}
y <- datasetvoice$label 
cbind(freq=table(y), percentage=prop.table(table(y))*100)
```

Desvio padrão.
```{r}
sapply(datasetvoice[,1:20], sd)
```

Skew.

```{r}
skew <- apply(datasetvoice[,1:20], 2, skewness)
print(skew)
```

Correlação.

```{r}
correlacao <- cor(datasetvoice[,1:20])
print(correlacao)
```

Histograma (univariado).


```{r fig.width = 10, fig.height = 10}
par(mfrow=c(5,4))
for(i in 1:20) {
  hist(datasetvoice[,i], main=names(datasetvoice)[i])
}
```



Gráfico de densidade (univariado).


```{r fig.width = 10, fig.height = 10}
par(mfrow=c(5,4))
for(i in 1:20) {
  plot(density(datasetvoice[,i]), main=names(datasetvoice)[i])
}
```


Boxplot e Whisker (univariado).


```{r fig.width = 10, fig.height = 10}
par(mfrow=c(5,4))
for(i in 1:20) {
  boxplot(datasetvoice[,i], main=names(datasetvoice)[i])
}
```


Gráfico de barras.

```{r fig.width = 10, fig.height = 10}
par(mfrow=c(5,4))
for(i in 1:20) {
  counts <- table(datasetvoice[,i])
  name <- names(datasetvoice)[i]
  barplot(counts, main=name)
}
```

Mapa de valores ausentes (univariado).
#
#```{r fig.width = 10, fig.height = 10}
#par(mfrow=c(1,1))
#datasetvoice(Soybean)
#missmap(Soybean, col=c("black", "grey"), legend=FALSE)
#```



Gráfico de correlação (multivariado)

```{r fig.width = 20, fig.height = 20}
correlacao <- cor(datasetvoice[,1:20])
cores <- colorRampPalette(c("red", "white", "blue"))
corrplot(correlacao, order="AOE", method="square", col=cores(20), tl.srt=45, tl.cex=0.75, tl.col="black")
corrplot(correlacao, add=TRUE, type="lower", method="number", order="AOE", col="black", diag=FALSE, tl.pos="n", cl.pos="n", number.cex=0.75)
```


Gráfico de dispersão (multivariado).


```{r fig.width = 15, fig.height = 15}
pairs(datasetvoice)
```


Gráfico de dispersão por classe (multivariado).


```{r fig.width = 15, fig.height = 15}
pairs(label~., data=datasetvoice, col=datasetvoice$label)
```

Gr?fico de densidade por classe (multivariado).

```{r fig.width = 15, fig.height = 10}
x <- datasetvoice[,1:20]
y <- datasetvoice[,21]
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)
```

Boxplot por classe (multivariado)
```{r fig.width = 15, fig.height = 10}
x <- datasetvoice[,1:20]
y <- datasetvoice[,21]
featurePlot(x=x, y=y, plot="box")
```











## Fim da analise
