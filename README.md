# ThirdPartyDataValidation
金融风控领域，第三方征信/评分数据有效性评价</br></br>
三率：</br></br>
覆盖率：全部指标至少有一个不为空的样本占比</br>
缺失率：单个变量的missing_rate</br>
准确率: 核验类变量“trigger”、“is_blacklist”跟公司内部的flag计算准确率</br></br>
三性：</br>
相关性（Corr）</br>
预测性（IV,Information Value）</br>
解释性 （BiVar图 必须单调且符合指标的业务意义）</br></br>
三度：</br></br>
区分度（KS、AUC）</br>
重要性（feature_importances）</br>
稳定度 (PSI)
