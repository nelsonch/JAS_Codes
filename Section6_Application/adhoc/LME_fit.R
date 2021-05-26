library(lme4)
library(MuMIn)
library(dplyr)
dat1 <- read.csv('real_application.csv', h=T)

###### scaling 
t1 <- subset(dat1, select = 'store')
t2 <- subset(dat1, select = c('marketing_5_1_after' , 'marketing_8_1_after', 
                              'marketing_3_1' , 'marketing_4_1' , 'marketing_6_1' ,
                               'marketing_6_2' , 'marketing_6_4' , 'marketing_6_6',
                              'marketing_6_7' , 'marketing_6_10', 'si' , 'unemployment_rate', 
                             'sales_qty'))
t22 <- scale(t2)
gg = cbind(t1, t22)
fm00 <- lmer(sales_qty ~ marketing_5_1_after + marketing_8_1_after 
             + marketing_3_1 + marketing_4_1 + marketing_6_1 +
               marketing_6_2 + marketing_6_4 + marketing_6_6
             + marketing_6_7 + marketing_6_10 + si + unemployment_rate + 
               (1|store), gg, REML = T)
print(r.squaredGLMM(fm00))
