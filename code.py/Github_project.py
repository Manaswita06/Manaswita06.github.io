#importing all the packages required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import math

#reading the dataset
df=pd.read_csv(r'C:\Users\HP\Downloads\Loan_dataset-Data_sheet.csv')
df
df.describe()       


plt.hist(df['INSTALLMENT'],bins=20)     #plotting histogram

#computation of alpha and beta using Method of Moments
x = np.array(df['INSTALLMENT'])
m1 = np.average(x)
ss = np.var(x)
print(m1)
print(ss)
alphaMM = m1*m1/ss         #computing alpha using method of moment estimate
betaMM = m1/ss             #computing beta using method of moment estimate
print(alphaMM)
print(betaMM)

#trying to fit gamma distribution to the histogram
fig,ax = plt.subplots(1,1)
ax.hist(x,density=True,bins=20)
xx = np.linspace(8.5,900,80)
ax.plot(xx, st.gamma.pdf(xx,alphaMM,scale=1/betaMM),"red",linestyle="dashed",label='gamma fit MM',lw=2)     #trying to find out if gamma distribution is the best fit for this histogram
plt.title("Gamma distribution fitting")
ax.legend(loc='best')
plt.show()
lm1 = np.average(np.log(x))
#Write the equation as a function
#digamma is the function Gamma'/Gamma
from scipy.special import digamma
fML = lambda a: (np.log(a) - digamma(a) - np.log(m1)+lm1)
fig, ax = plt.subplots(1,1)

xx = np.linspace(8.5,900,70)
ax.plot(xx,fML(xx))
ax.grid(True)
plt.show()

#finding alpha and beta using Maximum Likelihood Estimate
#For solving numerically, we will use scipy.optimize
import scipy.optimize as sopt
sol = sopt.root_scalar(fML, bracket=[5,2])
sol.root

alphaML = sol.root
betaML = alphaML/m1
print([alphaML, betaML])

#comparing MM Na ML estimate
fig,ax = plt.subplots(1,1)
ax.hist(x,density=True,bins=20)
xx = np.linspace(8.5,900,70)
ax.plot(xx, st.gamma.pdf(xx,alphaMM,scale=1/betaMM),lw='4',alpha=0.95,label='gamma fit MM',color = 'brown')
ax.plot(xx, st.gamma.pdf(xx,alphaML,scale=1/betaML),lw='2',label='gamma fit ML',color = 'orange')
ax.legend(loc='best')
plt.show()

#performing hypothesis testing
sd_installment = 207.06/math.sqrt(500)    #population standard deviation
alpha = 0.05                              #significance level
null_mean = 320                           #null hypothesis
c = (sd_installment*st.norm.ppf(0.05)) + null_mean      #computing critical value
s_mean, s_sd = np.average(x), np.sqrt(np.var(x,ddof=1))  #sample mean and sample standard deviation
print(f'Sample mean: {s_mean}, Sample standard deviation: {s_sd}, Critical value: {c} ')

#computing p-value and checking if we can accept or reject the null hypothesis
p_value = 1 - st.norm.cdf(abs(s_mean - null_mean))
if(p_value <  alpha):
  print("Reject Null Hypothesis")
else:
  print("Fail to Reject NUll Hypothesis")
  
#checking if sample mean is less than critical value, then reject the null hypothesis, else, accept the null hypothesis  
if s_mean < c:
  print("Reject Null Hypothesis")
else:
  print("Fail to Reject NUll Hypothesis")  
