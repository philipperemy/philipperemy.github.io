---
layout: post
title: Create the best cocktails with Deep Learning

excerpt: We will use Natural Language Processing and Recurrent Neural Networks to find which combinations of ingredients are likely to be the favorites of your guests !
---

Most cocktails aren't a big commitment to make, if you're a cocktail fan of any sort. You've got the supplies, a well-stocked bar and the know-how to mix them. There are some cocktails, though, that take far more effort than the average Margarita or Martini (though getting even these drinks just right is no small feat). We're talking dozens of ingredients, complicated home-made tinctures and infusions and fire. Could Artificial Intelligence help us to come up with some cocktails that we know, without any doubts, that our guests would fall in love with them?

The answer is yes. Now let's figure out how!

![png](/images/cocktails_1.jpg){: .center-image }

# Getting the data

One of the hardest tasks was to build a data set for supervised learning that contains the compositions of the cocktails along with the reviews and ratings. The only suitable website I could find was [1001cocktails.com](http://www.1001cocktails.com). This website is edited in French and does not provide an English translation. But don't worry, I will translate all the terms I use, as much as I can. They don't provide the data via API so I used the so called [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) library.
When you browse on a random cocktail, you will see a description like this:

![png](/images/cocktails_2.png){: .center-image }

For information, here is the English version:

**Mojito - Rating: 4.2/5 (5200 reviews)** 

Composition for one person:

- 6 cl of Cuban Rum
- 3 cl of Green Lemon juice
- 7 leaves of Mint
- Sparkling water (Perrier, Salvetat)
- 2 cl of cane sugar syrup

The second task is to write a parser to retrieve this structured information. I omit the code for clarity (I will upload the code at the end of this article).
From the source, we could successfully build a dataset of 4506 rows `(Name, RatingValue, RatingCount, Description)`. The quantity, quantifier and ingredient compose the description. The Mojito example would translate to:

```
['Mojito', 
4.2, 
5193, 
[['6.0', 'cl de', 'rhum cubain'], 
['3.0', 'cl de', 'jus de citrons verts'], 
['7.0', 'feuilles de', 'menthe'], 
['None', None, 'eau gazeuse (perrier, salvetat)'], 
['2.0', 'cl de', 'sirop de sucre de canne']]]
```

Let's derive some statistics based on our dataset. The distributions of the ratings exhibits a bell shape. It is negatively skewed because it has a long left tail. The distribution does seem to have a kurtosis comparable to the univariate normal distribution. The kurtosis of any univariate normal distribution is 3. It is common to compare the kurtosis of a distribution to this value. Distributions with kurtosis less than 3 are said to be platykurtic, although this does not imply the distribution is "flat-topped" as sometimes reported. Rather, it means the distribution produces fewer and less extreme outliers than does the normal distribution. An example of a platykurtic distribution is the uniform distribution, which does not produce outliers.
On the contrary, higher kurtosis means more of the variance is the result of infrequent extreme deviations, as opposed to frequent modestly sized deviations.

![png](/images/cocktails_3.png){: .center-image }

![png](/images/cocktails_4.png){: .center-image }

{% highlight python %}
import scipy.stats as ss
ratings = np.array([float(v[1]) for v in recipes])
print(ss.kurtosis(ratings))
print(ss.skew(ratings))

>> 2.79341883868
>> -0.664659407593
{% endhighlight %}

Note that it is always preferable to have a distribution as close as possible to the Uniform distribution. Otherwise, the dataset becomes unbalanced and the extremes are not correctly modeled.
One attempt to solve this issue is to convert the dataset into binary classification (very good cocktails vs not very good cocktails) and make sure that each class contains roughly the same number of samples.

The mean of the ratings is `3.373`. When we split the cocktails into two classes, `bad : rating < 3.373` and `good : rating >= 3.373`, we obtain a nearly perfect distribution :

- good : 0.503
- bad : 0.497

The french punctuations are also removed by running some SED command.
The dataset is split into a training set and a testing set with the respective proportions `3/4` and `1/4` (with equal proportions of `good` and `bad` in each).

We are now going to use Natural Language Processing and Deep Learning on this binary dataset to detect interesting features.

# Which are the best ingredients to mix together?

In this section, we're going to address this difficult problem: 

*If you had to choose 2, 3 or 4 ingredients and you want to maximize your chances to please your guests with your cocktails, what would you choose?*

We address this problem by using the [Natural Language Toolkit](http://www.nltk.org), especially Naive Bayes, and some scripts that [Andy Bromberg](http://andybromberg.com/) used for Sentiment Analysis on movies. If you're not already familiar with Naive Bayes, there's a very good StackOverflow article [here](http://stackoverflow.com/questions/10059594/a-simple-explanation-of-naive-bayes-classification). Very briefly, Naive Bayes uses the Bayes theorem and the prior information that we have to compute some posterior distributions and infer the class from there.
This method is very basic and usually serves as a baseline to compare models.

## One ingredient used to discriminate

Let's consider first the case with only one ingredient used to discriminate the cocktails. We put aside the quantities and focus solely on either accepting or rejecting a cocktail based on one of his ingredients.

From our dataset, we keep only the ingredients. For example, one cocktail is encoded as:

`vodka sirop_cassis sirop_caramel curacao_bleu eau`

For information, in english, it is: `vodka black_currant_syrup caramel_syrup curacao water`

Here is the output (after translation to English):

```
evaluating the best 200 word features
train on 3378 instances, test on 1128 instances
accuracy: 0.605
Most Informative Features
                sparkling_wine neg : pos    =     12.5 : 1.0
                   maple_syrup pos : neg    =      8.2 : 1.0
                  vanilla_pods neg : pos    =      6.4 : 1.0
                          wine neg : pos    =      5.4 : 1.0
                     chocolate pos : neg    =      5.2 : 1.0
                     rose_wine neg : pos    =      5.1 : 1.0
                 jasmine_syrup pos : neg    =      4.9 : 1.0
    caribbean_syrup_rum_flavor neg : pos    =      4.4 : 1.0
               Angelica_liquor neg : pos    =      4.4 : 1.0
                   Peach_syrup neg : pos    =      4.3 : 1.0
```

With one discriminating ingredient, we have an accuracy of 60% on our testing set. This output is very interesting and we can already derive some rules.

- If the cocktail contains sparkling wine, it is `12.5` times more likely to be considered below average (rating < `3.373/5`) by your guests.
- On the other hand, if you start to use maple syrup, you are likely to succeed `8.2` times more to please your guests.
- We can conclude that wine and cocktails are not a good match.
- Surprisingly, adding some chocolate seems to have a very positive effect. Our dataset, generated from [1001cocktails.com](http://www.1001cocktails.com), contains a few recipes such as `hot chocolate` or `punch chocolat`. This could bias positively the results. Some of the cocktails are alcohol-free.

Let's keep in mind that using one ingredient doesn't make a cocktail taste good. It's more about how the ingredients mix together that matters. However, we are more likely to reject a cocktail if it contains an inappropriate ingredient, such as wine for example.

## Discriminating with all the ingredients

Like we did before, we do not consider the quantities here and focus on the ingredients used in the cocktails. The trick to consider the interaction of multiple ingredients together is to use the ensemble combinations.

*What is a combination?*

Let's consider three elements `A`, `B` and `C`. The combination set is given by `A`, `B`, `C`, `A+B`, `A+C`, `B+C`. Let's apply this rule for:

`vodka lemon_juice sugar_syrup`

The output is:

`vodka lemon_juice sugar_syrup lemon_juice+sugar_syrup lemon_juice+vodka sugar_syrup+vodka lemon_juice+sugar_syrup+vodka`

A positive advantage of using this trick resides in the fact that we don't have to update our implementations. All what we need to do is read the dataset, generate the combinations for each cocktail, dump them to the corresponding files and call our NLP scripts for discrimination.
The term `lemon_juice+sugar_syrup+vodka` models the interaction of `lemon_juice`, `sugar_syrup` and `vodka` together. Let's re-run the NLP algorithm with the new dataset made of all the combinations for each cocktail.

**After evaluation, we figure out that the accuracy has increased from 60.5% to almost 71%. We now have the proof that combining the right ingredients together is the key to a good cocktail.** Let's analyze the results!

**And the winner is...**

![png](/images/cocktails_6.jpg){: .center-image }

{: style="color:gray; font-size: 300%; text-align: center;"}
**Vodka, Lemon, Sugar**

There are plenty of cocktails containing this awesome mix:

- Porticio bay 3.5
- Black Lime 4.1
- Mickado 3.5
- Lemon drop 3.6
- Huntsman 4.5
- Adios Motherfucker 3.9
- Astronaut 3.4
- Springtime cooler 3.8
- Endy's 4.3
- Liqueur de noix (Nocino) 5.0
- Makseb 4.8
- Floc me up 4.0
- Vodka sour 3.7
- Citrus twist 4.3
- A la Russe 3.5
- Vodka sling 4.4
- 1,2,3 sunlight! 3.7
- Marche Orientale 4.0
- Bikino 4.4
- Roulette Russe 3.6
- Vodka gimlet 4.1
- Impress Me 4.0
- Vodka Nikolaschka 3.8
- Belphégor 3.8
- Vin de citron 3.4
- Grenouille 5.0
- Granité Vodka Citron 3.9
- Kalachnikov 3.4
- Meojifo 3.7
- Bolshoï punch 3.5
- The Funny Super Green 3.8
- Vodka Cooler 4.0
- Magnum citron 4.0
- Air gunner 3.4
- Sandra in High Spirits 4.6

The number next to each cocktail's name is its rating (out of 5.0). When the cocktail contains at least some 
vodka, sugar, and lemon, its score is around `3.954`, much higher than the average which, I recall, is around `3.373`. 
One of the most famous listed cocktail above is the **Vodka Sour**, whose recipe is:

**Vodka Sour**

- 1 1/2 oz vodka
- 1/4 - 3/4 oz sugar syrup
- 3/4 oz lemon juice

Shake well over ice cubes in a shaker, and strain into a whiskey sour glass. Garnish with a stemmed cherry, and serve. Most of the reviews for this cocktail are actually very laudatory:

![png](/images/cocktails_5.png){: .center-image }

Some other great winners are:

- Maple Syrup (8.2 : 1.0)
- Raspberry and Sugar (7.6 : 1.0)
- Vodka and Cherry brandy (5.6 : 1.0)
- Cherry brandy + Triple Sec (4.9 : 1.0)
- Kiwi liquor (4.9 : 1.0)
- Orange juice and lychee liquor ( 4.9 : 1.0)
- Apricot liquor and vermouth (4.9 : 1.0)
- Gin, lemon juice, orange juice (4.5 : 1.0)
- Lemon juice, vodka, grenadine juice (4.3 : 1.0)
- Gin and Midori (4.3 : 1.0)
- Banana (3.1 : 1.0)
- Coconut liquor, grenadine syrup and vodka (3.0 : 1.0)
- Milk, Kahlua (Coffee liquor) and vodka (3.0 : 1.0)
- Angostura bitters and lemonade (3.0 : 1.0)
- Pineapple juice, banana liquor and vodka (3.0 : 1.0)
- Pineapple juice, coconut milk and rum (2.6 : 1.0) - **Pina Colada !!**
- Orange juice and raspberry syrup (2.6 : 1.0)

Don't be mistaken here. Having good mix of ingredients is important, however you must keep in mind that this is necessary but not sufficient to have a good score. Think of mixing a Vodka sour with wine to be convinced!
Now that we have listed some of the best ingredients you can add to make great cocktails, let's see the mixes you must avoid absolutely - I named the top losers.

**And the loser is...**

![png](/images/cocktails_7.jpg){: .center-image }

{: style="color:gray; font-size: 300%; text-align: center;"}
**Sparkling wine**

No need to say more. If a good friend of yours tells you something like: *Oh my god, I have a great idea for the drinks! Let's mix it with sparkling wine. Just take a deep breath, and say NO WAY please!*

Some other great losers are:

- Mixing Grapefruit juice and Rum (9.1 : 1.0) - try it if you're not convinced!
- Mixing Champaign and Strawberry liquor (7.1 : 1.0)
- Mixing Rum and Vanilla (5.7 : 1.0)
- Mixing Cognac and Red Porto (5.7 : 1.0)
- Sparkling water and Cuban Rum (5.7 : 1.0) - , what??
- Champaign, Cognac and Triple Sec, all together (5.1 : 1.0)
- 90 percent alcohol and water (4.7 : 1.0) - really I'm not kidding, it was in the dataset for real...

Again, keep in mind that a cocktail is seen as a loser if its rating is below the average of `3.373`. For example, a cocktail with `3.2` will be considered badly despite the fact that this is still considered as a good score among the guests. The nature of the distribution of the ratings forces us to make this choice.


Finally and for your information, the top 60 features to discriminate the cocktails are shown below (in French, I'm sorry in advance):

**Scroll down for the last words**

```
evaluating best 8000 word features
train on 3378 instances, test on 1128 instances
accuracy: 0.709
Most Informative Features
            vin_mousseux                neg : pos    =     12.5 : 1.0
jus_pamplemousses+rhum_blanc            neg : pos    =      9.1 : 1.0
           sirop_derable                pos : neg    =      8.2 : 1.0
        framboises+sucre                pos : neg    =      7.6 : 1.0
           chocolat+lait                pos : neg    =      7.6 : 1.0
champagne+liqueur_fraises               neg : pos    =      7.1 : 1.0
     citrons+sucre+vodka                pos : neg    =      6.9 : 1.0
         gousses_vanille                neg : pos    =      6.4 : 1.0
      rhum_blanc+vanille                neg : pos    =      5.7 : 1.0
      cognac+porto_rouge                neg : pos    =      5.7 : 1.0
eau_gazeuse+rhum_cubain                 neg : pos    =      5.7 : 1.0
creme_fraiche_liquide+sirop_grenadine   pos : neg    =      5.6 : 1.0
liqueur_cerises_(cherry_brandy)+vodka   pos : neg    =      5.6 : 1.0
                     vin                neg : pos    =      5.4 : 1.0
                chocolat                pos : neg    =      5.2 : 1.0
eau_gazeuse_(perrier,_salvetat)+menthe  neg : pos    =      5.1 : 1.0
champagne+cognac+triple_sec             neg : pos    =      5.1 : 1.0
                vin_rose                neg : pos    =      5.1 : 1.0
citrons_vert+rhum_cubain                neg : pos    =      5.1 : 1.0
      oranges+rhum_blanc                neg : pos    =      5.1 : 1.0
rhum_blanc+sirop_dorgeat                neg : pos    =      5.1 : 1.0
     cannelle+sirop_miel                neg : pos    =      5.1 : 1.0
       citrons+gingembre                neg : pos    =      5.1 : 1.0
citrons_verts+eau_gazeuse               neg : pos    =      5.1 : 1.0
liqueur_cerises+triple_sec              pos : neg    =      4.9 : 1.0
jus_doranges+liqueur_lychees            pos : neg    =      4.9 : 1.0
            sirop_jasmin                pos : neg    =      4.9 : 1.0
           liqueur_kiwis                pos : neg    =      4.9 : 1.0
liqueur_dabricots+vermouth_blanc        pos : neg    =      4.9 : 1.0
jus_citrons+sirop_dorgeat               pos : neg    =      4.9 : 1.0
        alcool_a_90o+eau                neg : pos    =      4.7 : 1.0
      menthe+rhum_cubain                neg : pos    =      4.7 : 1.0
gin+jus_citrons+jus_doranges            pos : neg    =      4.5 : 1.0
 chocolat_en_poudre+lait                pos : neg    =      4.5 : 1.0
     creme_bananes+vodkat               pos : neg    =      4.5 : 1.0
      alcool_a_90o+sucret               neg : pos    =      4.5 : 1.0
sirop_caribbean_(saveur_rhum)           neg : pos    =      4.4 : 1.0
cognac+liqueur_cerises                  neg : pos    =      4.4 : 1.0
               angelique                neg : pos    =      4.4 : 1.0
cannelle+noix_muscade_rapee             neg : pos    =      4.4 : 1.0
jus_citrons+jus_doranges+sucre_canne    neg : pos    =      4.4 : 1.0
         cannelle+cognac                neg : pos    =      4.4 : 1.0
cognac+porto_rouge+sucre                neg : pos    =      4.4 : 1.0
   cannelle+cognac+sucre                neg : pos    =      4.4 : 1.0
citrons_vert+menthe+rhum_cubain         neg : pos    =      4.4 : 1.0
jus_dananas+jus_doranges+pamplemousses  neg : pos    =      4.4 : 1.0
    cafe+creme_chantilly                neg : pos    =      4.4 : 1.0
        jus_pommes+vodka                pos : neg    =      4.4 : 1.0
creme_whisky_(baileys)+lait             pos : neg    =      4.3 : 1.0
  jus_dananas+jus_pommes                pos : neg    =      4.3 : 1.0
jus_citrons+sirop_grenadine+vodka       pos : neg    =      4.3 : 1.0
            citron+vodka                pos : neg    =      4.3 : 1.0
              gin+midori                pos : neg    =      4.3 : 1.0
jus_citrons+sirop_framboises            pos : neg    =      4.3 : 1.0
                 guarana                pos : neg    =      4.3 : 1.0
          eau+framboises                pos : neg    =      4.3 : 1.0
            chocolat+eau                pos : neg    =      4.3 : 1.0
             sirop_peche                neg : pos    =      4.3 : 1.0
         citrons+oranges                neg : pos    =      4.3 : 1.0
noix_muscade_rapee+sucre                neg : pos    =      3.7 : 1.0
```

And last but not least, Barack Obama once said, *I've rarely met a cocktail I didn't like*. Please, don't disappoint him if he ever shows in one your parties!

Also, always keep in mind that:

![png](/images/cocktails_8.jpg){: .center-image }

Thanks for reading.