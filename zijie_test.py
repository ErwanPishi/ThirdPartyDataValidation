#%% prelimilaries
import re
import toad
import datetime
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family']='SimHei'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from toad.plot import bin_plot
from sklearn.model_selection import train_test_split
from toad.metrics import KS,PSI,KS_bucket #单独的KS不是分箱后的KS，
                                          # 实际当中应当采用KS_bucket分箱后进行操作
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score as rec_score
from sklearn.metrics import accuracy_score as acc_score
from sklearn.metrics import roc_auc_score as auc
savepath = 'C:/Users/zhaoyiming/Desktop/字节测试/'

def str2date(date_str,sep):
    '''
    Parameters
    ----------
    str : TYPE
        DESCRIPTION.
    sep : - /
        连接符.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    year = datetime.date(*map(int, date_str.split(sep))).year
    month = datetime.date(*map(int, date_str.split(sep))).month
    return str(year)+'-'+str(month)

def plot_dist(df,col):
    sns.distplot(df[col],kde=True,bins=20,rug=True)
    
def cover_rate(df):
    #至少有一个非空就算覆盖到
    df=df.copy()
    if 'flag' in df.columns:
        df.drop(columns=['flag'],inplace=True)
    length = df.shape[0]
    notna_rate_by_rows = df.notna().mean(axis=1)   #按行算非空比例
    cover_r = notna_rate_by_rows[notna_rate_by_rows>0].size/length #非空比例大于0则覆盖
    return cover_r

def missing_rate(pinggu,y_col):
    missing_rate = pinggu.isnull().sum()/pinggu.shape[0]
    missing_rate.rename('缺失率',inplace=True)
    missing_rate.to_excel(savepath+y_col+'/missing_rate.xlsx')
    

def cal_psi(pinggu,y_col):
    '''
    原理：选择最早的那一个周期的样本作为base
    '''
    Combiner = toad.transform.Combiner()
    Combiner.fit(pinggu,y=pinggu[y_col],
                 n_bins=10,
                 method='quantile',
                 empty_separate=True)
    base_mon = pinggu['appl_dt_min'].min()

    use_col = pinggu.columns.drop('appl_dt_min')
    psi_df = toad.metrics.PSI(test=pinggu[~pinggu.appl_dt_min.isin([base_mon])][use_col],
                              base=pinggu[pinggu.appl_dt_min==base_mon][use_col],
                              combiner=Combiner)
    psi_df = psi_df.to_frame(name='psi')
    psi_df.to_csv(savepath+y_col+'/psi.csv')
    return psi_df

def cal_iv(pinggu,y_col):
    savepath = 'C:/Users/zhaoyiming/Desktop/字节测试/'
    iv_df = toad.quality(pinggu, target=y_col, iv_only=True)[["iv"]]
    iv_df = iv_df.sort_values(by = 'iv',ascending=False)
    
    if len(iv_df)>=10:
        plt.figure(figsize=[48,36])
        iv_df[:10].plot(kind='bar')
        plt.xticks(rotation=60)
        plt.title('变量IV值前10展示图')
        plt.savefig(savepath+y_col+'/iv.png',
                    bbox_inches='tight',
                    dpi=300)
        plt.show()
        
    else:
        plt.figure(figsize=[48,36])
        iv_df.plot(kind='bar')
        plt.xticks(rotation=60)
        plt.title('变量IV值展示图')
        plt.savefig(savepath+y_col+'/iv.png',
                    bbox_inches='tight',
                    dpi=600)
        plt.show()
        
    iv_df.to_csv(savepath+y_col+'/iv.csv')
      
    return iv_df

def plot_bivar_and_cal_ks_value(pinggu,method,y_col):
    '''
    Parameters
    ----------
    subfolder : str
        文档路径.
    pinggu : pd.DataFrame
        不用剔flag.
    method : str
        'dt': decision tree,
        'quantile': 等频,
        'kmean',
        'step':等步长.

    Outputs
    -------
    bivar图，KS值，变量分布图.

    '''
    ks_lst = [ ]
    sns.set(font_scale=1)
    savepath = 'C:/Users/zhaoyiming/Desktop/字节测试/'
    cols = pinggu.columns.drop(y_col)
    for col in cols:
        c = toad.transform.Combiner()
        for bins in range(4, 7):
            c.fit(pinggu,
                  y=y_col,
                  method=method,
                  n_bins=bins,
                  empty_separate=True)  # ,n_bins=4~6
            bin_plot(c.transform(pinggu[[col, y_col]],
                                 labels=True),
                     x=col, target=y_col)
            name = col+'_'+'bins'+str(bins)
            plt.savefig(savepath+y_col+'/'+name+'.png')
            plt.show()
            
        plot_dist(pinggu,col)
        plt.savefig(savepath+y_col+'/'+col+'_dist.png')
        plt.show()
        
        ks_bucket = KS_bucket(pinggu[col], pinggu[y_col])
        ks = abs(ks_bucket.ks).max()
        ks_lst.append(ks)
        print(col, " ks: ", ks)
    ks_lst = pd.Series(ks_lst, index=cols)
    ks_lst.sort_values(ascending=False,inplace=True)
    ks_lst.to_csv(savepath+y_col+'/ks.csv')
    
    print('DONE')
    
def evaluate(pinggu,y_col):
    '''
    输出变量的IV、KS、CORR、distplot、missing_rate
    '''
    
    #相关性
    savepath = 'C:/Users/zhaoyiming/Desktop/字节测试/'
    tmp_df = df.iloc[:,~df.columns.isin([y_col])]
    plt.figure(figsize=[48,36])
    sns.set(font_scale=9) 
    sns.heatmap(tmp_df.corr(),cmap='YlGnBu') #Yellow Green Blue
    plt.savefig(savepath+y_col+'/corr.png',dpi=300)
    plt.show()
    #缺失率
    missing_rate = pinggu.isnull().sum()/pinggu.shape[0]
    missing_rate.to_csv(savepath+y_col+'/missing_rate.csv')
    
    #iv ks 
    for col in pinggu:
        plot_bivar_and_cal_ks_value(pinggu,
                                    method='quantile',
                                    y_col=y_col)
        cal_iv(pinggu,y_col=y_col)
        
def my_style_converter(corr_matrix):
    '''
    Description:
        
    将相关系数矩阵转化为my_style
    
    Example:
        
       corr    dual
    1  0.8   ('a','b')
    2  0.7   ('c','d')
    
    Parameters
    ----------
    corr_matrix
    
    Returns
    -------
    my_style.
    
    
    
    '''
    var_list=corr_matrix.columns
    length = len(var_list)
    
    
    corr = [ ]
    dual = [ ]
    for i in range(0,length):
        for j in range(i+1,length):
            corr.append(corr_matrix.loc[var_list[i],var_list[j]])
            dual.append((var_list[i],var_list[j]))
            
    my_style = pd.DataFrame({'Corr':corr,'dual':dual})   
    my_style.sort_values(by='Corr',ascending=False,inplace=True)
    my_style.reset_index(drop=True,inplace=True)
    return my_style



def cal_ks(pinggu,var_name,label,bucket=10,method='quantile'):
    '''
    Parameters
    ----------
    pinggu : DataFrame
        不含Y.
    var_name : str
        要计算ks的变量名.
    label : iterable
        标签列.

    Returns
    -------
    ks : float
        返回变量分箱后的KS值.

    '''
    # pinggu不含Y
    ks_bucket = KS_bucket(pinggu[var_name],label,
                          bucket,method)
    ks = abs(ks_bucket.ks).max()
    return ks


def correlation_elimator(pinggu,label,
                         method='ks',
                         threshold=0.7):
    '''
    Parameters
    ----------
    label : iterable
        标签列.
    pinggu : DataFrame
        不含Y.
        
    Methodology
    -----------
      1.计算相关系数矩阵，并转化为my_style (按照相关系数降序排列了)

      2.从my_style中保留相关系数大于等于threshold的行，
      并计算此时的行数rows
        3.挑选出相关系数最高的变量A和B
        4.分别计算KS_a KS_b
        5.剔除KS较小的那个，从my_style中删除这一行 
        6.计数君+1，记录剔除的变量,更新my_style
      7.重复3-6，如果计数君==rows或者my_style已经被筛成空集了,则结束循环

    Returns
    -------
    Series of the name of filtered variables.
    '''
    corr_matrix = pinggu.corr()
    my_style = my_style_converter(corr_matrix)
    my_style = my_style[my_style.Corr>=threshold]
    
    rows = len(my_style)
    
    drop_lst = [ ] 
    count = 0
    dual_col = pd.Series(my_style.dual)
    
    while ( (count<rows) and (len(dual_col)>0) ):
        (m,n) = dual_col.head(1)[0]
        KS_m = cal_ks(pinggu,m,label)
        KS_n = cal_ks(pinggu,n,label)
        if (KS_m >= KS_n):
            drop_lst.append(n)
            drop_var = n
        else:
            drop_lst.append(m)
            drop_var = m
            
        True_lst = [ ]
        for (a,b) in dual_col:
            print(a,b)
            if ( (drop_var==a) or (drop_var==b) ):
                True_lst.append(False)
            else:
                True_lst.append(True)
            
        
        dual_col=dual_col[True_lst]
        dual_col.reset_index(drop=True,inplace=True)
        count+=1

    return drop_lst


#%% load_data
path1 = 'C:/Users/zhaoyiming/Desktop/字节测试'
sample = pd.read_excel(path1+'/sample.xlsx')

#标签
label_data = pd.read_csv(path1+'/label.csv', encoding="gbk")

#有重复项 要DROP_DUPLICATES
label_data.drop_duplicates(keep='first',
                           inplace=True)
#去除无关变量
label_data.drop(columns=['cust_id',
                    'ovrd_days_max_y',
                    'cust_cn_nm',
                    'cust_mpl_phn_no'],
                inplace=True)

'''
fpd7:首期还款逾期 first payment deliquency  首逾指标 首逾指标常常与欺诈相关
     首笔需要还款的账单（也就是第一期） 在最后还款日后7天内未还款且未办理延期的客户
dpd30: 逾期30天 记为M1
'''   
label_data.isnull( ).sum( ) #无nan值


drop_lst = sample.columns.drop(['appl_dt_min',
                                'crtf_num',
                                'cust_cn_nm',
                                'cust_mpl_phn_no'])
sample.isnull( ).sum( )
sample.dropna(how='all',
              subset=drop_lst,
              inplace=True)

#%% dpd30_y
data = pd.merge(left=label_data,
                right=sample,
                on=['crtf_num','appl_dt_min'],
                how='left')

data.drop(columns=['appl_dt_min',
                   'crtf_num','cust_cn_nm',
                   'cust_mpl_phn_no'],
          inplace=True)
# 缺失率
missing_rate(data, 'dpd30_y')

data = pd.merge(left=label_data,
                right=sample,
                on=['crtf_num','appl_dt_min'], #一个客户一个日期算作一条记录
                how='inner')

# 相关性
tmp_df = sample.iloc[:,~sample.columns.isin(['fpd7',
                                     'appl_dt_min',
                                     'dpd30_y'])]
plt.figure(figsize=[48,36])
sns.set(font_scale=9) 
sns.heatmap(tmp_df.corr(),cmap='YlGnBu')
plt.savefig(savepath+'dpd30_y/corr.png',dpi=300)
plt.show()
# 消除相关性
drop_lst = correlation_elimator(label_data['dpd30_y'], tmp_df)

# 更正为标准的日期格式
data.appl_dt_min=data.appl_dt_min.apply(lambda x : str2date(x,'/'))

data.drop(columns=['crtf_num','cust_cn_nm',
                   'cust_mpl_phn_no'],
          inplace=True)

data['appl_dt_min'].value_counts().sort_index()

# evaluate
df = data.copy()
df.drop(columns=['fpd7'],inplace=True)

y = 'dpd30_y'
df.drop(columns=['appl_dt_min'],inplace=True)
evaluate(df,y)

# plot PSI
df_psi = cal_psi(df,y_col='dpd30_y')
df_psi=df_psi[2:]
tmp = pd.cut(df_psi.psi,#分箱数据
        bins = [0,0.05,0.1,0.25,1],#分箱断点
        right = False,
        labels=['0-0.05','0.05-0.1','0.1-0.25','>0.25'])# 分箱后分类
plt.title('PSI分布')
tmp.value_counts().sort_index().plot(kind='bar')
plt.xticks(rotation=360)
plt.savefig(savepath+'psi.png',dpi=600)
plt.show()

# IV
df_iv = cal_iv(df.drop(columns='fpd7'),y_col='dpd30_y')
tmp = pd.cut(df_iv.iv,#分箱数据
        bins = [0,0.02,0.05,0.1,0.3,0.5,0.7,1],#分箱断点
        right = False,
        labels=['0-0.02','0.02-0.05','0.05-0.1',
                '0.1-0.3','0.3-0.5','0.5-0.7',
                '>0.7'])# 分箱后分类
plt.title('变量IV值分布')
tmp.value_counts().sort_index().plot(kind='bar')
plt.xticks(rotation=360)
plt.savefig(savepath+'变量iv值分布.png',dpi=600)
plt.show()

# 样本直方图
plt.title('有效样本分布')
df.appl_dt_min.value_counts().sort_index().plot(kind='bar')
plt.xticks(rotation=360)
plt.savefig(savepath+'distribution.png',dpi=600)
plt.show()


# 模型初测 存疑 没用WOE编码
xtr,xte,ytr,yte = train_test_split(df.drop(columns=['dpd30_y']),
                                   df['dpd30_y'],
                                   test_size=0.3,
                                   random_state=0,
                                   stratify=df['dpd30_y'])

import warnings
warnings.filterwarnings('ignore')

model = xgb.XGBClassifier(n_estimators=33)
model.fit(xtr,ytr)

pred_proba = model.predict_proba(xte)[:,1]
print(auc(yte,pred_proba)) #69.51
print(KS(pred_proba,yte))  #34.32








