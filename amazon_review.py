# RatingProduct & SortingReviewsin Amazon

#İş Problemi

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış
# sonrası verilen puanların doğru şekilde hesaplanmasıdır. Bu
# problemin çözümü e-ticaret sitesi için daha fazla müşteri
# memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer
# problem ise ürünlere verilen yorumların doğru bir şekilde
# sıralanması olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne
# çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem
# maddi kayıp hem de müşteri kaybına neden olacaktır. Bu 2 temel
# problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını
# arttırırken müşteriler ise satın alma yolculuğunu sorunsuz olarak
# tamamlayacaktır.

# Veri Seti Hikayesi

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# reviewerID Kullanıcı ID’si
# asin Ürün ID’si
    # reviewerName Kullanıcı Adı
# helpful Faydalı değerlendirme derecesi
# reviewText Değerlendirme
# overall Ürün rating’i
# summary Değerlendirme özeti
# unixReviewTime Değerlendirme zamanı
# reviewTime Değerlendirme zamanı Raw
# day_diff Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes Değerlendirmenin faydalı bulunma sayısı
# total_vote Değerlendirmeye verilen oy sayısı

# Ilk olarak ilgili kutuphanelerimizi import edelim.

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as st

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# csv dosyamizi okutalimm

df = pd.read_csv('C:/Users/pc/PycharmProjects/pythonProject2/4.Hafta/amazon_review.csv')
df_ = df.copy()
df.head()
df_.head()
# Veri setimize bir bakis atalim

def  first_look_at_data (dataframe , head = 10):
    print('####### shape #######')
    print(dataframe.shape)


    print('######## types ######')
    print(dataframe.dtypes)


    print('########## NA #######')
    print(dataframe.isnull().sum())


    print('#### statistics #####')
    print(dataframe.describe().T)

    print('####value_counts#####')
    print(df.value_counts())

    print('##### quantiles #####')
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

df.describe().T
first_look_at_data(df , head=10)

# Tarih degeri iceren degiskenlerin tipini date yapalim.

df['reviewTime'] = df['reviewTime'].astype('datetime64[ns]')

df.head()





# Average Rating’i güncel yorumlara göre hesaplayalim ve var olan average rating ile kıyaslayalim.

#Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır. Bu görevde amacımız verilen
# puanları tarihe göre ağırlıklandırarak değerlendirmek. İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı
# puanın karşılaştırılması gerekmektedir.

# Ürünün ortalama puanını hesaplayalim.

df['overall'].mean()

# Tarihe göre ağırlıklı puan ortalamasını hesaplayacagiz.

# Ilk olarak 'reviewTime' degiskeninin tipini date yapalim.

df['reviewTime'] = df['reviewTime'].astype('datetime64[ns]')

# Analiz tarihini bugunun tarihi degerlendirmenin yapildigi son tarih yapalim.

current_date = df["reviewTime"].max()

# her bir puan-yorum tarihi ile current_date'in farkını gün cinsinden ifade ederek yeni değişken oluşturalim ve gün
# cinsinden ifade edilen değişkeni quantile fonksiyonu ile 4'e bölüp (3 çeyrek verilirse 4 parça çıkar) çeyrekliklerden
# gelen değerlere göre ağırlıklandırma yapalim.

df['difference'] = (current_date - df['reviewTime']).dt.days
df['difference'].dtype
df.head()

df['difference'].quantile([.25 , .50 , .75])

# ceyreklik degerlerimiz asagidaki gibidir.

# 0.25000   280.00000
# 0.50000   430.00000
# 0.75000   600.00000

print(df.loc[(df['difference'] <= 280) , 'overall'].mean())
print(df.loc[(df['difference'] > 280) & (df['difference'] <= 430) , 'overall'].mean())
print(df.loc[(df['difference'] > 430) & (df['difference'] <= 600) , 'overall'].mean())
print(df.loc[(df['difference'] > 600) , 'overall'].mean())

# 4.6957928802588995
# 4.636140637775961
# 4.571661237785016
# 4.4462540716612375

# Zamanla oy ortalamasinin yukseldigini goruyoruz.

# Simdi zamana bagli olarak bir agirlikli ortalama alabilecegimiz bir fonksiyon yazalim.

def weighted_sorting_score(dataframe , w1 = 28 , w2 = 26 , w3 = 24 , w4 = 22):
    return dataframe.loc[(dataframe['difference'] <= 280), 'overall'].mean() * w1 / 100 + \
        dataframe.loc[(dataframe['difference'] > 280) & (df['difference'] <= 430) , 'overall'].mean() * w2 /100 + \
        dataframe.loc[(df['difference'] > 430) & (df['difference'] <= 600) , 'overall'].mean() * w3 / 100 + \
        dataframe.loc[(df['difference'] > 600) , 'overall'].mean() * w4 / 100


weighted_sorting_score(df)

# Zamana gore agirlikli ortalamamiz 4.595593165128118 olarak hesaplanmistir.



# Up - Down Difference Score , Average Rating , Wilson Lower Bound Score yontemlerini calisalim. Ve son olarak
# Wilson Lower Bound Score yontemine gore ilk 20 yorumu belirleyelim.


# helpful_no degiskenini uretelim.

df['helpful_no'] = df['total_vote'] - df['helpful_yes']


#############################

# Up - Down Difference Score

#############################


# Faydali bulunan yorumlardan, faydali bulunmayan yorumlari cikaralim. Bunu bir fonksiyon tanimlayarak yapip,
# sonra bu fonksiyonu veri setimize ekleyelim

def score_pos_neg_diff(up , down):
    return up - down

df['score_pos_neg_diff'] = score_pos_neg_diff(df['helpful_yes'] , df['helpful_no'])

df.sort_values('score_pos_neg_diff' , ascending=False).head(20)

# up - down difference score yontemine gore ilk 20 gozlemi siraladigimizda faydali bulunan yorumlarin tum yorumlara olan
# oraninin daha yuksek oldugu bazi noktalarda, siralamasinin daha asagida oldugundan,
# Bu yontemin yaniltici oldugu gorulmektedir.


######################

# Score Average Rating

######################

# Faydali bulunan yorumlarin sayisinin tum yorumlara gore oranina bakiyoruz. Bunu bir fonksiyon tanimlayarak yapip,
# sonra bu fonksiyonu veri setimize ekleyelim.

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df['score_average_rating'] = df.apply(lambda x: score_average_rating(x['helpful_yes'] , x['helpful_no']) , axis=1)

df.head()

df.sort_values('score_average_rating' , ascending=False).head(20)

# score_average_rating yontemine gore en yuksek orana sahip 20 gozleme baktigimizda, basari oranlarinin %100 olmasina
# ragmen frekans degerlerinin 1 oldugunu , bununda degerlendirme acisindan oldukca yaniltici oldugunu gormekteyiz.


##########################

# Wilson Lower Bound Score

##########################


# Up - Down Difference Score ve Score Average Rating yontemlerinin hangi sebeplerden oturu yaniltici olabilecegini
# gozlemledik. Dolayisiyla gerek frekans gerekse de oran gibi degerleri goz ardi etmeden dogru siralamayi
# yapabilecegimiz Wilson Lower Bound Score yontemini uygulayacagiz.

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.head()

df.sort_values('wilson_lower_bound' , ascending=False).head(20)

# Wilson Lower Bound Score yontemine gore bir siralama yaptigimizda goruyoruz ki, frekans ve ortalama degerleri dikkate
# alinmis ve siralama cok daha dogru bir sekilde ortaya cikmistir.