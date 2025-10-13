library(lsa)
library(reticulate)
library(fpc)
library(R.utils)
use_python("~/miniconda/envs/myenvR/bin/python", required = TRUE)

np <- import("numpy")
source('helper.r')

args = as.character(commandArgs(trailingOnly = T))
nargs = length(args)

task = args[1]
key = args[2]


# WBT
WBT1 <- function(x,cl,P,s,vv) {
  result <- tryCatch({
    Indices.WBT(x,cl,P,s,vv)$ccc
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}

WBT <- function(x,cl,P,s,vv) {
  result <- tryCatch({
    withTimeout({WBT1(x,cl,P,s,vv)}, timeout = 3600)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}



# SDbw1
SDbw1<-function(x, cl) {
  result <- tryCatch({
    Index.SDbw(x, cl)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}

SDbw<-function(x, cl) {
  result <- tryCatch({
    withTimeout({SDbw1(x, cl)}, timeout = 3600)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}


library(callr)
CDbw <- function(x, cl) {
  result <- tryCatch({
    r(
      function(x, cl) {
        library(fpc)
        cdbw(x, cl)$cdbw
      },
      args = list(x = x, cl = cl),
      timeout = 3600   # seconds
    )
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}



# dunn
Dunn1<-function(md, cl, Data, method) {
  result <- tryCatch({
    Index.dunn(md, cl, Data, method)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}
Dunn<-function(md, cl, Data, method) {
  result <- tryCatch({
    withTimeout({Dunn1(md, cl, Data, method)}, timeout = 3600)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}


# cindex
#Indice.cindex(d=md, cl=cl1)
Cindex1<-function(d,cl) {
  result <- tryCatch({
    Indice.cindex(d, cl)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}
Cindex<-function(d,cl) {
  result <- tryCatch({
    withTimeout({Cindex1(d,cl)}, timeout = 3600)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}


file = paste0(task,'/raw_tmp/key_',key, '.npz')
data = np$load(file)


jeu=data$f[["jeu"]]
print(dim(jeu))
TT=data$f[["TT"]]
md=data$f[["md"]]
cmd=data$f[["cmd"]]
labelset = data$f[['labelset']]


if (grepl('COIL-100', key)){
  is_cal_ccc <- FALSE
}else{
  is_cal_ccc <- TRUE
  nn <- dim(jeu)[1]
  sizeEigenTT <- length(eigen(TT)$value)
  eigenValues <- eigen(TT/(nn-1))$value
  for (i in 1:sizeEigenTT) 
  {
    if (eigenValues[i] < 0) {
      is_cal_ccc <- FALSE # The TSS matrix should be indefinite for ccc calcuaation
    } 
  }
}
print(is_cal_ccc)
# match label to make it 1-indexed
cl1 = labelset
unique_cl1 = unique(cl1)
indices = match(cl1, unique_cl1)
cl1 = indices
print(unique(cl1))

if (is_cal_ccc){
    s1 <- sqrt(eigenValues)
    ss <- rep(1,sizeEigenTT)
    for (i in 1:sizeEigenTT) 
    {
      if (s1[i]!=0) 
        ss[i]=s1[i]
    }
    vv <- prod(ss)  
    ccc = WBT(x=jeu, cl=cl1, P=TT,s=ss,vv=vv) #max
}else{
    print('skip ccc calculation')
    ccc=NA
}


print('ccc')
dunn = Dunn(md, cl1, Data=jeu, method=NULL) #max #depend on md matrix
print('dunn')
cind = - Cindex(d=md, cl=cl1) #min #depend on md matrix
print('cind')
sdbw = - SDbw(jeu, cl1) #min # need data
print('sdbw')
ccdbw = CDbw(jeu, cl1) #max # need data
print('cdbw')
print(ccdbw)
print('done')

folder_path = paste0(task,"/raw_tmp")
if (!dir.exists(folder_path)) {
  dir.create(folder_path)
}
np$savez(paste0(task,"/raw_tmp/rr_", key, ".npz"), 
  ccc=ccc,
  dunn=dunn, cind=cind,
  sdbw=sdbw, ccdbw=ccdbw,
  models=args)
