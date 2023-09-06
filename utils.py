import pandas as pd
import numpy
from scipy import stats
import math

def check_missing_values(file):
    check = numpy.where(file.isnull())
    check = pd.DataFrame(check)
    for i in range(len(check.iloc[0,:])):
        print(f'missing value in the row {check.iloc[0,i]} of the column {check.iloc[1,i]}.')
    print(f'The total number of missing values is: {len(check.axes[1])}')

def find_outliers(dataframe,x,coef):
    '''
    The function to find outliers in numerical variables on the basis of IQR rule
    '''
    count_lower = []    
    count_upper = []
    Q1 = dataframe.iloc[:,x].quantile(0.25)
    Q3 = dataframe.iloc[:,x].quantile(0.75)
    IQR = Q3 - Q1
    lower_lim = Q1 - coef*IQR
    upper_lim = Q3 + coef*IQR
    for data in range(len(dataframe.iloc[:,0])):
        if dataframe.iloc[data,x] < lower_lim:
            count_lower.append(data)
        elif dataframe.iloc[data,x] > upper_lim:
            count_upper.append(data)
    print(f'The number of lower outliers is:{len(count_lower)},\
    The number of upper outliers is :{len(count_upper)}')

def transform_data(data,name):
    data = data.set_index(name)
    data = (data.iloc[:,:]/data.iloc[:,:].sum()*100).round(2)
    return data

def transform_data_reverse(data,name):
    data=data.set_index(name)
    data.iloc[:,0]= (data.iloc[:,0]/data.sum(axis=1)*100).round(2)
    data.iloc[:,1] = (data.iloc[:,1]/data.sum(axis=1)*100).round(2)
    return data

def get_percent(data):
    data = (data.iloc[:,:]/data.iloc[:,:].sum())
    return data

def plot_bars(data,y,y_label,title):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=data.index, y=y,
            data=data, 
            errorbar="sd", color = sns.color_palette()[0]).set(title=title)
    plt.xticks(rotation=90)
    ax.bar_label(ax.containers[0])
    ax.set_ylabel(y_label)
    ax.set_xlabel("")
    plt.show()

def stacked_bars(file, title_label, title):
    ax = file.plot(kind="barh", stacked=True, rot=0)
    ax.legend(title=title_label, bbox_to_anchor=(1, 1.02),
             loc='upper left')
    plt.xlabel("")
    plt.xticks(rotation = "vertical")
    plt.xlabel("%")
    for c in ax.containers:
        ax.bar_label(c, label_type='center')
    plt.title(title)
    plt.show()

def chi_square_test(data):
    stat, p, dof, expected = chi2_contingency(data)
    alpha = 0.05
    print(f'Pearson chi square test:{stat.round(3)}')
    print('')
    print(f'P_value: {p.round(3)}')
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (H0 holds true)')

def kruscal_wallis_test(data1, data2, data3, data4, data5):
    stat, p = stats.kruskal(data1, data2, data3, data4, data5)
    print(f"Kruscal Wallis test:")
    print('')
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
    print('')

def permutation_test_period(file1, file2, column, sample, period1, period2):
    reviews_s_all = list(file1[column]) + list(file2[column])
    diff_observed = abs(file1[column][file1[column]=='POSITIVE'].count() - \
                        file2[column][file2[column]=='POSITIVE'].count())

    n_repeats = 10000
    random_differences = []
    for i in range(n_repeats):
        random.shuffle(reviews_s_all)
        reviews_new1 = reviews_s_all[:1000]
        reviews_new2 = reviews_s_all[1000:]
        new_difference = abs(len([num for num in reviews_new1 if num == 'POSITIVE']) - \
        len([num for num in reviews_new2 if num == 'POSITIVE']))
        random_differences.append(new_difference)

    n_greater_equal = 0
    for i in range(n_repeats):
        if random_differences[i] >= diff_observed:
            n_greater_equal = n_greater_equal + 1

    p_value = n_greater_equal/n_repeats
    
    if p_value > 0.05:
        print(f'Hypothesis H0 that samples <{sample[period1]}> and <{sample[period2]}> \
are from the same distribution is approved.\np_value: {p_value}')
        print('')
    else:
        print(f'Hypothesis H0 that samples <{sample[period1]}> and <{sample[period2]}> \
are from the same distribution is not approved.\n The alternative hypothesis holds.\np_value: {p_value}')
        print('')

def permutation_test(file1, file2, column, sample):
    reviews_s_all = list(file1[column]) + list(file2[column])
    diff_observed = abs(file1[column][file1[column]=='POSITIVE'].count() - \
                        file2[column][file2[column]=='POSITIVE'].count())

    n_repeats = 10000
    random_differences = []
    for i in range(n_repeats):
        random.shuffle(reviews_s_all)
        reviews_new1 = reviews_s_all[:1000]
        reviews_new2 = reviews_s_all[1000:]
        new_difference = abs(len([num for num in reviews_new1 if num == 'POSITIVE']) - \
        len([num for num in reviews_new2 if num == 'POSITIVE']))
        random_differences.append(new_difference)

    n_greater_equal = 0
    for i in range(n_repeats):
        if random_differences[i] >= diff_observed:
            n_greater_equal = n_greater_equal + 1

    p_value = n_greater_equal/n_repeats
    
    if p_value > 0.05:
        print(f'Hypothesis H0 that samples <{file1[sample].iat[0]}> and <{file2[sample].iat[0]}> \
are from the same distribution is approved.\np_value: {p_value}')
        print('')
    else:
        print(f'Hypothesis H0 that samples <{file1[sample].iat[0]}> and <{file2[sample].iat[0]}> \
are from the same distribution is not approved.\n The alternative hypothesis holds.\np_value: {p_value}')
        print('')

def mean_confidence_interval(data, column, confidence):
    a = 1.0 * numpy.array(data[column])
    n = len(a)
    m, se = numpy.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1+confidence)/2., n-1)
    print(f'Mean: {m.round(3)}; Confidence interval: {(m-h).round(3)}, {(m+h).round(3)};')
    print(f'Confidence: {confidence*100} %; Standard error: {se.round(3)}; t-value: {(stats.t.ppf((1+confidence)/2., n-1)).round(3)}')

def confidence_interval_proportion(file, column, value, sample):
    reviews_s6_ci = proportion_confint(count=file[column][file[column]==value].count(),
                   nobs=file[column].count(),  
                   alpha=(1 - 0.95))
    print(f'Confidence intervals for number of {value} in the <label> variable of \
<{file[sample].iat[0]}> sample: \n{reviews_s6_ci}')

def confidence_interval_proportion_period(file, column, value, sample, period):
    reviews_s6_ci = proportion_confint(count=file[column][file[column]==value].count(),
                   nobs=file[column].count(),  
                   alpha=(1 - 0.95))
    print(f'Confidence intervals for number of positives in the <label> variable of \
<{sample[period]}> sample: \n{reviews_s6_ci}')

def confidence_intervals_difference(file1,file2, n, column, value, sample, confidence):
    p1=(file1[column][file1[column]==value].count())/n
    p2=(file2[column][file2[column]==value].count())/n
    upper = (p1-p2) + stats.norm.ppf(q=1-(1 - confidence)/2)*math.sqrt((p1*(1-p1)/n + p2*(1-p2)/n))
    lower = (p1-p2) - stats.norm.ppf(q=1-(1 - confidence)/2)*math.sqrt((p1*(1-p1)/n + p2*(1-p2)/n))
    print(f'There is a {confidence*100} chance that the confidence interval of {lower.round(3), upper.round(3)} \
contains the true difference \nin the proportion of {value} sentiments between reviews \
for podcasts in <{file1[sample].iat[0]}> and <{file2[sample].iat[0]}> categories.')
    print('')
    if (lower < 0 and upper < 0) or (lower > 0 and upper > 0):
        print(f'Since this interval does not contain the value “0” it means \n\
that it’s highly likely that there is a true difference in the proportion \n\
of {value} sentiments between reviews for podcasts \n in <{file1.iloc[0,6]}> and <{file2.iloc[0,6]}> categories.')
        print('')
    else:
        print(f'Since this interval contains the value “0” it means that it’s unlikely \n\
that there is a true difference in the proportion of {value} \
sentiments between reviews for podcasts \n in <{file1[sample].iat[0]}> and <{file2[sample].iat[0]}> categories.')
        print('')

def confidence_intervals_difference_period(file1,file2, n, column, value, sample, period1, period2, confidence):
    p1=(file1[column][file1[column]==value].count())/n
    p2=(file2[column][file2[column]==value].count())/n
    upper = (p1-p2) + stats.norm.ppf(q=1-(1 - confidence)/2)*math.sqrt((p1*(1-p1)/n + p2*(1-p2)/n))
    lower = (p1-p2) - stats.norm.ppf(q=1-(1 - confidence)/2)*math.sqrt((p1*(1-p1)/n + p2*(1-p2)/n))
    print(f'There is a {confidence*100} % chance that the confidence interval of {lower.round(3), upper.round(3)} \
contains the true difference \nin the proportion of {value} sentiments between reviews \n\
for podcasts in <{sample[period1]}> and <{sample[period2]}> time periods.')
    print('')
    if (lower < 0 and upper < 0) or (lower > 0 and upper > 0):
        print(f'Since this interval does not contain the value “0” it means \n\
that it’s highly likely that there is a true difference in the proportion \n\
of {value} sentiments between reviews for podcasts \n in <{sample[period1]}> and \
<{sample[period2]}> time periods.')
        print('')
    else:
        print(f'Since this interval contains the value “0” it means that it’s unlikely \n\
that there is a true difference in the proportion of {value} \
sentiments between reviews for podcasts \n in <{sample[period1]}> and <{sample[period2]}> time periods.')
        print('')

def sample_reviews_sentiments(file, sample):
    data = file.sample(n=sample)
    sentiments = []
    contents = []
    titles = []
    podcasts_id = []
    authors_id = []
    ratings = []
    created_ats = []
    categories = []
    for i in range(len(data)):  
        podcast_id = data.iloc[i,0]
        title = data.iloc[i,1]
        content = data.iloc[i,2][:1000]
        rating = data.iloc[i,3]
        author_id = data.iloc[i,4]
        created_at = data.iloc[i,5]
        category = data.iloc[i,6]
        sentiment = sentiment_pipeline(content)
        sentiments.append(sentiment[0])
        podcasts_id.append(podcast_id)
        titles.append(title)
        contents.append(content)
        ratings.append(rating)
        authors_id.append(author_id)
        created_ats.append(created_at)
        categories.append(category)
    sentiments_pd = pd.DataFrame(sentiments)
    podcasts_id_pd = pd.DataFrame({'podcast_id':podcasts_id})
    titles_pd = pd.DataFrame({'title':titles})
    contents_pd = pd.DataFrame({'content':contents})
    ratings_pd = pd.DataFrame({'rating':ratings})
    authors_id_pd = pd.DataFrame({'author_id':authors_id})
    created_ats_pd = pd.DataFrame({'created_at':created_ats})
    categories_pd = pd.DataFrame({'category':categories})
    sentiment_reviews = pd.concat([podcasts_id_pd, titles_pd, 
                                   contents_pd,ratings_pd, 
                                   authors_id_pd, 
                                   created_ats_pd,
                                   categories_pd,
                                   sentiments_pd],axis=1)
    return sentiment_reviews

def find_inputs(model, x_data, y_data): 
    data = {}
    var_list = [0,1,2,3,4,5,6,7,8,9,10,11]
    i_list = [[]]
    u = 0
    while u < 10000:
        i = random.choices(var_list[:-1], k=random.choice(var_list[1:]))
        i = list(numpy.unique(i))
        x = x_data[:,i]
        mod = model.fit(x, y_data)
        r_sq = mod.score(x, y_data)
        if i not in i_list:
            data[f"{i}"] = r_sq
            i_list.append(i)
        u+=1
    print('Combinations of independent variables for the model with the highest coeficient of determination:')
    print(f'{max(data, key=data.get)}')
    print('Coeficient of determination:') 
    print(data[f'{max(data, key=data.get)}'])
    return data





