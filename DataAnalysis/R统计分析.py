# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from rpy2.robjects import r

# <codecell>

a = r['matrix'](range(10), nrow = 2)
print a

# <codecell>

load("C:\\Code\\RData\\Basic1.RData")
x<- c(4,466,67,47)
x[x<220&&x>10]
x[x<220&x>10]
y= runif(100,min=0,max=1)
y
mean(y>0.5)
x[is.na(x)]
attach(mtcars)
mtcar2<- data.frame(mtcars[,c(1,4)])
mtcar2
save(mtcar2,'')
getwd()
setwd('C:\Code\RData')
setwd('C:\\Code\\RData')
save(mtcar2,'MyR.Rdata')
open('a')
aa
x
y
sunflowerplot(x)
x<- c(1:10)
x
x<- c(1:100)
sunflowerplot(x,x^2)
x<- c(1:10)
sunflowerplot(x,x^2)
pie(x)
interaction.plot(x,2*x^2,x^3)
interaction.plot(x,2*x^2,x^3)
y<-x
y
aab
aa
qqplot(x)
qqplot(x,x^2)
text(x, y, expression(p==over(1,1+e^-(beta*x+alpha))))
exit
text(x, y, expression(p==over(1,1+e^-(beta*x+alpha))))
text(x, y, as.expression(substitute(R^2==r,
list(r=round(Rsquared, 3)))))
plot(x)
plot(x,x^3)
text(x, y, expression(p==over(1,1+e^-(beta*x+alpha))))
layout(matrix(1:3,3,1))
data<-read.table('Swal.dat')
sample(1;55，2)
sample(1:55，2)
sample(1:55, 2)
binom(1,0.3)
unif(3,4)
layout(matrix(1:3, 3, 1))
layout(matrix(1:3, 3, 1))
op<- par(mfrow=c(2,2))
oo
op
help.start()
n<- 20
k<-seq(0,n)
n<- 20
p<- 0.2
k<-seq(0,n)
plot(k,dbinom(k,n,p),type='h',main='Binomial distribution, n=20,p=0/2',xlab='k')
savehistory(file ='.Rhistory')
loadhistory(file='1.Rhistory')
load("C:\\Code\\RData\\Basic1.RData")
x<- c(4,466,67,47)
x[x<220&&x>10]
x[x<220&x>10]
y= runif(100,min=0,max=1)
y
mean(y>0.5)
x[is.na(x)]
attach(mtcars)
mtcar2<- data.frame(mtcars[,c(1,4)])
mtcar2
save(mtcar2,'')
getwd()
setwd('C:\Code\RData')
setwd('C:\\Code\\RData')
save(mtcar2,'MyR.Rdata')
open('a')
aa
x
y
sunflowerplot(x)
x<- c(1:10)
x
x<- c(1:100)
sunflowerplot(x,x^2)
x<- c(1:10)
sunflowerplot(x,x^2)
pie(x)
interaction.plot(x,2*x^2,x^3)
interaction.plot(x,2*x^2,x^3)
y<-x
y
aab
aa
qqplot(x)
qqplot(x,x^2)
text(x, y, expression(p==over(1,1+e^-(beta*x+alpha))))
exit
text(x, y, expression(p==over(1,1+e^-(beta*x+alpha))))
text(x, y, as.expression(substitute(R^2==r,
list(r=round(Rsquared, 3)))))
plot(x)
plot(x,x^3)
text(x, y, expression(p==over(1,1+e^-(beta*x+alpha))))
layout(matrix(1:3,3,1))
data<-read.table('Swal.dat')
sample(1;55，2)
sample(1:55，2)
sample(1:55, 2)
binom(1,0.3)
unif(3,4)
layout(matrix(1:3, 3, 1))
layout(matrix(1:3, 3, 1))
op<- par(mfrow=c(2,2))
oo
op
help.start()
n<- 20
k<-seq(0,n)
n<- 20
p<- 0.2
k<-seq(0,n)
plot(k,dbinom(k,n,p),type='h',main='Binomial distribution, n=20,p=0/2',xlab='k')
savehistory(file ='.Rhistory')
open(file='1.Rhistory')
loadhistory(file='*.Rhistory')
lambda<-04
lambda<-0.4
k<-seq(0,20)
k
plot(k,dpois(k,lambda),type='h',main='Geometric distribution,p=0.5',xlabel='k')
plot(k,dpois(k,lambda),type='h',main='Geometric distribution,p=0.5',xlabeli=k')
plot(k,dpois(k,lambda),type='h',main='Geometric distribution,p=0.5',xlabeli=k')
plot(k,dpois(k,lambda),type='h',main='Geometric distribution,p=0.5',xlabxxhhk')
plot(k,dpois(k,lambda),type='h',main='Geometric distribution,p=0.5',xlabxxhhk')
plot(k,dpois(k,lambda),type='h',main='Geometric distribution,p=0.5',xlab="k")
plot(k,dpois(k,lambda),type='h',main='Geometric distribution,p=0.5',xlab="k")
plot(k,dpois(k,lambda),type='h',main='Geometric distribution,p=0.5',xlab="k")
plot(k,dgeom(k,p),type='h',main='Geometric distribution,p=0.5',xlab="k")
library(DAAG)
library(DAAG)
chooseCRANmirror()
utils:::menuInstallPkgs()
library(DA
)
data()
library(DAAG)
install.packages("DAAG")
library(DAAG)
library("DAAG")
data(possum)
fpossum<-possum[possum$sex=="f",]\
fpossum<-possum[possum$sex=="f",]
fpossum
length(fpossum)
possum
?stem
dens<-density(totl
)
Eye.Hair <- matrix(c(68,20,15,5, 119,84,54,29,
26,17,14,14, 7,94,10,16), nrow=4,byrow=T)
colnames(Eye.Hair)<-c('Brown','Bule','Hazel','Green')
rownames(Eye.Hair)<-c('Black','Brown','Red","Blond")
)
colnames(Eye.Hair)<-c('Brown','Bule','Hazel','Green')
rownames(Eye.Hair)<-c('Black','Brown','Red','Blond')
Eye.Hair
install.packages("ISwR")
library('juul')
table(Eye.Hair)
Eye.Hair
margin.table(Eye.Hair,1)
data()
data()
data(HairEyeColor)
HairEyeColor
a<-as.table(apply(HairEyeColor,c(1,2),sum)
)
a
apply(HairEyeColor,c(1,2))
c(1,2)
?apply
a<-as.table(apply(HairEyeColor,1,sum)
)
a
a<-as.table(apply(HairEyeColor,c(1,2),sum))
a
barplot(a,legend.text=attr(a,'dimnames')$Hair)
a<-as.table(apply(HairEyeColor,2,sum)
)
a
a<-as.table(apply(HairEyeColor,c(2,1),sum))
barplot(a,legend.text=attr(a,'dimnames')$Hair)
a
dotplot(Eye.Hair)
dotchart(E
ye.Hair)
dotchart(Eye.Hair)
mydata<-read.delim("clipborad")
mydata<-read.delim("clipborad")
mydata<-read.delim("clipborad")
X<-c(1,1,0 ,1 ,0, 0, 1, 0 ,1 ,1,1, 0 ,1, 1 ,0 ,1,
0 ,0 ,1, 0 ,1 ,0,1, 0 ,0 ,1,1 ,0 ,1, 1, 0, 1)
theta<-mean(X)
theta
t<-theta/(1-theta)
t
t
f<-function(P)(p^517)*(1-p)^483
f
optimize(f,c(0,1),maximum=TRUE)
optimize(f,c(0,1))
f<-function(p)(p^517)*(1-p)^483
optimize(f,c(0,1),maximum=TRUE)
x<-c(1.21, 1.30, 1.39, 1.42, 1.47, 1.56, 1.68, 1.72, 1.98, 2.10)
y<-c(3.90, 4.50, 4.20, 4.83, 4.16, 4.93, 4.32, 4.99, 4.70, 5.20)
plot(x,y)
cor.test(x,y)
x<-c(318, 910, 200, 409, 415, 502, 314, 1210, 1022, 1225)
y<-c(524, 1019, 638, 815, 913, 928, 605, 1516, 1219, 1624)
lm.reg<- lm(y~1+x)
lm.reg
?summary
summary(lm.reg)
confint(lm.reg,level=0.95)
summary(lm.reg)
plot(lm.reg)
plot(lm.reg)
plot(lm.reg)
point <- data.frame(x=415)
lm.pred<- predict(lm.reg,point,interval='prediction',level=0.95)
lm.pred
y<-c(162, 120, 223, 131, 67, 169, 81, 192, 116, 55,
252, 232, 144, 103, 212)
x1<-c(274, 180, 375, 205, 86, 265, 98, 330, 195, 53,
430, 372, 236, 157, 370)
x2<-c(2450, 3250, 3802, 2838, 2347, 3782, 3008, 2450,
2137, 2560, 4020, 4427, 2660, 2088, 2605)
scales <- data.frame(y,x1,x2)
scales
lm.reg<- lm(y~x1+x2,data = scales)
lm.reg
summary(lm.reg)
plot(lm.reg)
plot(lm.reg)

