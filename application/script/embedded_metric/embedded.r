library(lsa)
library(reticulate)
library(fpc)
library(R.utils)

np <- import("numpy")
source('helper.r')

args = as.character(commandArgs(trailingOnly = T))
nargs = length(args)

task = args[1]
m = args[2]
key = args[3]



WBT <- function(x,cl,P,s,vv) {
  result <- tryCatch({
    Indices.WBT(x,cl,P,s,vv)$ccc
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}

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
    withTimeout({SDbw1(x, cl)}, timeout = 120)
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
      timeout = 120   # seconds
    )
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}


file = paste0(task,'/tmp/',m,'.npz')
print(file)
data = np$load(file)
jeu=data$f[["jeu"]]
print(dim(jeu))
TT=data$f[["TT"]]
md=data$f[["md"]]
cmd=data$f[["cmd"]]

is_cal_ccc <- TRUE
nn <- dim(jeu)[1]
sizeEigenTT <- length(eigen(TT)$value)
eigenValues <- eigen(TT/(nn-1))$value
for (i in 1:sizeEigenTT) 
{
  if (eigenValues[i] < 0) {
    is_cal_ccc <- FALSE
  } 
}


file = paste0(task,'/tmp/',m,'_', key, '.npz')
print(file)
data = np$load(file)
labelset = data$f[['labelset']]

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

dunn = Index.dunn(md, cl1, Data=jeu, method=NULL) #max
print('dunn')
cind = - Indice.cindex(d=md, cl=cl1) #min
print('cind')
sdbw = - SDbw(jeu, cl1) #min
print('sdbw')
ccdbw = CDbw(jeu, cl1) #max
print('cdbw')
  
print('done')


folder_path = paste0(task,"/tmp")
if (!dir.exists(folder_path)) {
  dir.create(folder_path)
}
np$savez(paste0(task,"/tmp/rr_", m, "_", key, ".npz"), ccc=ccc,
  dunn=dunn, cind=cind,
  sdbw=sdbw, ccdbw=ccdbw,
  models=args)
