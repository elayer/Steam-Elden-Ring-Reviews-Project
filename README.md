## Elden Ring Reviews NLP Project - Overview:

* Scraped roughly two hundred thousand reviews from Steam on the game <i>'Elden Ring'</i>. 

* Tokenized the review text to conduct N-Gram analysis, create word clouds, and construct data to be fed into NLP models (namely Sentiment Analysis).

* To perform Sentiment Classification, I began model building by using Naive Bayes, SGD Classifier and Logistic Regression. Following this, I built a deep learning PyTorch model utilizing HuggingFace transformers. Here, I used the RoBERTa model.

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
Some noteable findings from performing exploratory data analysis can be seen below. I found from looking at the Bi-Grams of the words in the reviews corpus, a lot of them primarily vaunted the game with a few exceptions mainly geared at pointing out performance issues. Similar sentiments can be seen in uni and tri-grams as well. The Chi2 influential term analysis graph is the most interesting to me. I found words that primarily distinguish between positive and negative reviews dealt with screen ultrawide support and performance issues such as crashes. The last picture looks at the LDA results chart, with one topic being comprised of positive comments. The negative topic comprises mainly words related to the performance of the game.

![alt text](https://github.com/elayer/Steam-Elden-Ring-Reviews-Project/blob/main/bigrams_picture.png "BiGrams Counts")
![alt text](https://github.com/elayer/Steam-Elden-Ring-Reviews-Project/blob/main/stylecloud.png "Word Stylecloud")
![alt text](https://github.com/elayer/Steam-Elden-Ring-Reviews-Project/blob/main/chi2_picture.png "Chi2 Influential Words")
![alt text](https://github.com/elayer/Steam-Elden-Ring-Reviews-Project/blob/main/lda_picture.png "LDA Topic Example")

## Model Building (Sentiment Classification)
Before building any models, I transformed the text using Tfidf Vectorizer and Count Vectorizer in order to make the data trainable. 

* I started model building with Naive Bayes. From here, confusion matrix results improved as I moved to using the SGD classifier, and then Logistic Regression. 

* I then attempted to use PyTorch with the HuggingFace Transformer library (namely, using RoBERTa) to maximize sentiment classification results. Although RoBERTA with PyTorch performed better than Logistic Regression, Logistic Regression achieved good results as well albeit the recall for non-recommended reviews being low. 


## Model Performance (Sentiment Classification)
The Naive Bayes, SGDClassifier, and Logistic Regression models resptively achieved improved results. I then built the PyTorch model with HuggingFace. Since training the entire model with PyTorch using just 4 Epochs and 5 folds for cross validation would have taken more than 4 days on my computer, I only used on epoch on one fold. After this, I gathered the results of the model based on only that much training.

<b>(The possible labels for classification here are 0 : Non-recommended and 1 : Recommended)</b>

Below are the Macro F1 Scores of each model built:

* Naive Bayes: 0.48

* SGD Classifier (SVM using Hinge Loss): 0.69

* Logistic Regression: 0.81

* RoBERTa with PyTorch: 0.87 (after only 1 epoch on 1 fold of data)

With a more powerful machine, I think we can achieve a robust model knowing the granular differences between recommended and non-recommended reviews. Here is an example of some predictions made from the model using a few samples from another fold that model wasn't trained on:

![alt text](https://github.com/elayer/Amazon-Computer-Project/blob/main/amazon_homepage.png "Website Homepage")

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
