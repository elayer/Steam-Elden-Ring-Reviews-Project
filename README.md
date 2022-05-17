## Elden Ring Reviews NLP Project - Overview:

* Scraped roughly two hundred thousand reviews from Steam on the game <i>'Elden Ring'</i>. 

* Tokenized the review text to conduct N-Gram analysis, create word clouds, and construct data to be fed into NLP models (namely Sentiment Analysis).

* Began model building by using Naive Bayes, SGD Classifier and Logistic Regression to perform Sentiment Analysis. Following this, I built a deep learning 
PyTorch model utilizing HuggingFace transformers. Here, I used the RoBERTa model.

* Lastly, to analyze the topics of discussion among the apps to track down potential areas of game improvement and reception of the game itself, I performed 
LDA (Latent Dirichlet Analysis) and LSA (Latent Semantic Analysis) to extract topic information and key distinguishing words in the text corpus.


## Code and Resources Used:

**Python Version:** 3.8.5

**Packages:** numpy, pandas, requests, beautiful soup, matplotlib, seaborn, sklearn, huggingface transformers, pytorch, nltk, gensim, spacy, re

**Web Framework Requirements Command:** ```pip install -r requirements.txt```

## References:

* Various project structure and process elements were learned from Ken Jee's YouTube series: 
https://www.youtube.com/watch?v=MpF9HENQjDo&list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

* Helpful information on how to scrap Steam reviews:
https://andrew-muller.medium.com/scraping-steam-user-reviews-9a43f9e38c92

* Elaborate and effective PyTorch structure and architecture nuances I learned in this Kaggle notebook from a competition I participated in to learn:
https://www.kaggle.com/code/yasufuminakama/nbme-deberta-base-baseline-train

* LSA in Python implementation guide: 
https://towardsdatascience.com/latent-semantic-analysis-sentiment-classification-with-python-5f657346f6a3

* LDA in Python implementation guide: 
https://towardsdatascience.com/topic-modelling-in-python-with-spacy-and-gensim-dc8f7748bdbf


## Web Scraping:

Created a web scraper using Requests and Beauitful Soup. Using the Steam scraper, I obtained the following information from each record of reviews (relevant to project):
*   Language
*   Review
*   Voted Up (Recommended)
*   Votes Up
*   Votes Funny
*   Weighted Vote Score (helpfulness)
*   Comment Count
*   Steam Purchase
*   Received for Free

## Data Cleaning

After collecting data, I performed several necessary text cleaning steps in order to analyze the corpus and perform EDA. I went through the following steps to clean and prepare the data:

* Loaded the spacy English corpus and updated the stop words list to include /n and /t

* With each review separated in a separate list, I lemmatized the text to keep only the root word and lowercased each word

* Then, I only kept words that were not punctuation and were either numeric or alphabetic characters of text

* Lastly, in order to maintain the integrity of the reviews, I dropped reviews that were less than 15 characters long to maintain reviews conducive to NLP algorithms. I also removed reviews more than 512 characters long for the PyTorch model to operate on the reviews correctly

## EDA
Some noteable findings from performing exploratory data analysis can be seen below. When going from a low to more high-end processor, the price of a computer does indeed increase. The same applies to RAM. In addition, I noticed some brands were priced higher even with similar or lower amounts of disk space. I eventually found that just as big of a driver in price was the brand of a computer, and not only the specs.

![alt text](https://github.com/elayer/Amazon-Computer-Project/blob/main/price-by-processor-type.png "Processor Type Boxplots")
![alt text](https://github.com/elayer/Amazon-Computer-Project/blob/main/price-histogram.png "Price Distribution")
![alt text](https://github.com/elayer/Amazon-Computer-Project/blob/main/price-by-RAM-boxplots.png "RAM Boxplots")
![alt text](https://github.com/elayer/Amazon-Computer-Project/blob/main/price-to-brand-lmplots.png "RAM vs. Price per Brand")

## Model Building
Before building any models, I transformed the categorical variables into appropriate numeric types. I transformed brand into dummy variables since some of the more expensive computers were similar in distribution of price, and the same goes for less expensive computers. I then ordinally encoded the processor types since each type seemed to have a steady increase in price as you improved the quality of the processor.

I first tried a few different linear models and some variations of them:

* Starting with Linear regression, and then trying Lasso, Ridge, and ElasticNet to see if the results would change since we have many binary columns. 

* This then led me to try Random Forest, XGBoost, and CatBoost regression because of the sparse binary/categorical nature of most of the attributes in the data. 

## Model Performance
The Random Forest, XGBoost, and CatBoost regression models respectively had improved performances. These models considerably outperformed the linear regression models I tried previously. Below are the R2 score values for the models:

* Linear Regression: 72.28 (the best of the linear models as a baseline)

* Random Forest Regression: 83.76

* XGBoost Regression: 85.38

* CatBoost Regression: 85.87

I used Optuna with XGBoost and CatBoost to build an optimized model especially with the various attributes that these algorithms have. Since the data used is very sparse or represent specific definitive values (categories), it makes sense that the tree-based methods perform much better. 

<b>UPDATE:</b> Even though the CatBoost regression obtained the highest r2 score, the linear regression model with regularization could still be a better choice. In the future, we could compare the above models and choose one based on mock predictions made from the Flask application.

## Productionization
I created a Flask API hosted on a local webserver. For this step I primarily followed the productionization step from the YouTube tutorial series found in the refernces above. This endpoint could be used to take in certain aspects of a computer, make appropriate transformations to the variables, and return a predicted price for a computer.  

<b>UPDATE:</b> A working local Flask API simulation is now uploaded and working. Below are a few sample pictures:

![alt text](https://github.com/elayer/Amazon-Computer-Project/blob/main/amazon_homepage.png "Website Homepage")
![alt text](https://github.com/elayer/Amazon-Computer-Project/blob/main/amazon_prediction.png "Website Prediction Page")
![alt text](https://github.com/elayer/Amazon-Computer-Project/blob/main/amazon_products_example.png "Products Example")

Even though the given specs provides a good prediction for some examples (see picture above), there are also product listings that have higher and lower prices which may skew some results when making predictions. For instance, there are computers with the exact the same specs as the examples listed above with much higher prices. In the future, this could be an area of improvement for this project.

## Future Improvements
If there are any efforts in the future to improve this project, I would start with the data itself. Though, it is very difficult to obtain data by scraping amazon product listings pages in a more sophisticated way to obtain more honest and elaborate data about the product listings. Some aspects that I believe could have helped the project is improved data quality and display on Amazon, and practical points in product listings to acquire more details that could benefit model construction such as a computer's GPU. 

**I also think there are still outliers that may throw off some predictions since even products with the same base specs could have varying prices due to some other factors we can't capture with the data at hand. I could choose to return to the data itself and remove any outliers from records with similar specs across the board yet differ in price greatly. See production example for further understanding.**

In terms of model building, perhaps some advanced regression methods such as stacking or blending methods could enhance the model beyond the metrics currently obtained. 

We also could juxtapose the other models with mock predictions made with the Flask application. Even though the regression models obtained lower r2 scores, they could still be more viable models, particularly the ones with regularization. 
