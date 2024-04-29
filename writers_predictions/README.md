# Think Like a Data Scientist - [Kaggle Text Classification Community Competition](https://www.kaggle.com/competitions/ise-competition-1)
# What are the competition goals?
1. Predict the author of excerpts from horror stories by Edgar Allan Poe, Mary Shelley and HP Lovecraft. The predictions are evaluted using logloss.
2. Learn and apply ensembling method (both optuna and stacking).
3. Make a comprehensive note of methods and ideas used in this competition so as write a medium post and also use the knowledge in the YouTube project I am working on.
4. Build a methodology roadmap for text classification projects going forward
5. Write all code by hand. No copy pasting.
6. Create helper functions that can be reused for future problems and update the helper_functions.py file.

# What are my assumptions about the data and the competition?
*The aims of listing my assumptions is to check if they hold true and where not verifable acknowledge my assumptions*
1. A combination of feature engineering and BERT will produce a score better than the current competition private score.
2. The text excerpts match the original text. There was no error in data gathering and if there is, it is consistent between train and test data.

# What is my guiding light for this competition?
Bloom's Taxonomy and Ultralearning.

Remember -> Understand -> Apply -> Analyse -> Evaluate -> Create

# Has someone does this before?
I will consider the novelty from the angle of this being a text classification problem and then the answer is yes, many people have done text classification before.

To enable me gather information, I will go with the following plan.
- [ ] Search Kaggle for 'text classification'.
- [ ] Identify a list of top 30 Discussions (Topics) sorted by relevance. Let's call this ***Route A***
- [ ] Identify a list of top 15 Datasets sorted by relevanve. Let's call this ***Route B***
- [ ] Identify a list of top 10 Competitions sorted by relevance. Let's call this ***Route C***

## Route A - Discussions
* Write down methods from the top discussions (if there is code screenshot and save)
* Read the comments to see if there are any extra nuggets of wisdom that can be gotten

## Route B - Datasets
* Sort Code in Dataset by 'Most Votes' and then 'Most Comments', get the top 10 code or all Gold posts (whichever is more) for each sort. When sorting by 'Most Comments' after 'Most Votes', if post already exist in the list, go to the next available post.
* Study the notebooks for interesting and useful ideas.
* Screenshot any useful code. Ensure to name the code snippets properly and save in the appropraite folder.

## Route C - Competitions
* Sort Code in Competition by 'Most Votes' then 'Public Score' then 'Most Comments. Get top 10 posts or all Gold posts (whichever is more) for each sort. Like in Datasets route, ensure there are no duplicates posts considered.
* Do same for Discussion tab sorting by 'Most Votes' and then 'Most Comments'.
* Ensure as much useful information and code is gathered.

## Route Combination
Based off all the knowledge gathered so far, create a flowchart on the plan for the project referencing code snippets or ideas as annotations where necessary.

After this is done, take a 1 day break to recover from what I assume has been a stressful 6 days and then commence the actual execution of the project.

During execution, since code must have come from different sources. All code will have to be rewritten to match my style and project structure. ChatGPT (and other GPTs) are allowed as well as good old googling and reading of books.

# Gathering of Route Lists
## ROUTE A - DISCUSSIONS
1. ~~[https://www.kaggle.com/competitions/quora-insincere-questions-classification/discussion/78928](https://www.kaggle.com/competitions/quora-insincere-questions-classification/discussion/78928)~~
2. [https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/discussion/95603](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/discussion/95603)
3. [https://www.kaggle.com/competitions/quora-insincere-questions-classification/discussion/70821](https://www.kaggle.com/competitions/quora-insincere-questions-classification/discussion/70821)
4. [https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/335896](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/335896)
5. [https://www.kaggle.com/competitions/google-quest-challenge/discussion/128184](https://www.kaggle.com/competitions/google-quest-challenge/discussion/128184)
6. [https://www.kaggle.com/competitions/quora-insincere-questions-classification/discussion/72519](https://www.kaggle.com/competitions/quora-insincere-questions-classification/discussion/72519)
7. [https://www.kaggle.com/discussions/general/205128](https://www.kaggle.com/discussions/general/205128)
8. [https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/494873](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/494873)
9. [https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/326998](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/326998)
10. [https://www.kaggle.com/competitions/quora-insincere-questions-classification/discussion/70825](https://www.kaggle.com/competitions/quora-insincere-questions-classification/discussion/70825)
11. [https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/361592](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/361592)
12. [https://www.kaggle.com/discussions/getting-started/100283](https://www.kaggle.com/discussions/getting-started/100283)
13. [https://www.kaggle.com/competitions/lshtc/discussion/6993](https://www.kaggle.com/competitions/lshtc/discussion/6993)
14. [https://www.kaggle.com/competitions/nlp-getting-started/discussion/344698](https://www.kaggle.com/competitions/nlp-getting-started/discussion/344698)
15. [https://www.kaggle.com/discussions/getting-started/412744](https://www.kaggle.com/discussions/getting-started/412744)
16. [https://www.kaggle.com/discussions/general/217877](https://www.kaggle.com/discussions/general/217877)
17. [https://www.kaggle.com/discussions/general/291426](https://www.kaggle.com/discussions/general/291426)
18. [https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/50477](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/50477)
19. [https://www.kaggle.com/discussions/getting-started/220493](https://www.kaggle.com/discussions/getting-started/220493)
20. [https://www.kaggle.com/discussions/general/197130](https://www.kaggle.com/discussions/general/197130)
21. [https://www.kaggle.com/discussions/questions-and-answers/237859](https://www.kaggle.com/discussions/questions-and-answers/237859)
22. [https://www.kaggle.com/discussions/general/342059](https://www.kaggle.com/discussions/general/342059)
23. [https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/463471](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/463471)
24. [https://www.kaggle.com/discussions/getting-started/435185](https://www.kaggle.com/discussions/getting-started/435185)
25. [https://www.kaggle.com/discussions/general/358858](https://www.kaggle.com/discussions/general/358858)
26. [https://www.kaggle.com/discussions/general/401615](https://www.kaggle.com/discussions/general/401615)
27. [https://www.kaggle.com/discussions/general/232212](https://www.kaggle.com/discussions/general/232212)
28. [https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/424942](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/424942)
29. [https://www.kaggle.com/discussions/general/431389](https://www.kaggle.com/discussions/general/431389)
30. [https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/discussion/142254](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/discussion/142254)


## ROUTE B - DATASETS
1. [Coronavirus tweets NLP - Text Classification](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification)
2. [Hierarchical text classification](https://www.kaggle.com/datasets/kashnitsky/hierarchical-text-classification)
3. [Ecommerce Text Classification](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification)
4. [Legal Citation Text Classification](https://www.kaggle.com/datasets/shivamb/legal-citation-text-classification)
5. [FastText crawl 300d 2M](https://www.kaggle.com/datasets/yekenot/fasttext-crawl-300d-2m/code?datasetId=14154&sortBy=voteCount)
6. [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
7. [Amazon reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)
8. [Emotion Detection from Text](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text)
9. [Genre Classification Dataset IMDb](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)
10. [News Headlines Dataset For Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
11. [Twitter Sentiment Analysis](https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech)
12. [Twitter and Reddit Sentimental analysis Dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset)
13. [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
14. [E-Mail classification NLP](https://www.kaggle.com/datasets/datatattle/email-classification-nlp)
15. [IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset/data)

*Bonus* 

16. [Physics vs Chemistry vs Biology](https://www.kaggle.com/datasets/vivmankar/physics-vs-chemistry-vs-biology)

## ROUTE C - COMPETITIONS
1. [Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
2. [Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification)
3. [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification)
4. [The Learning Agency Lab - PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data)
5. [What's Cooking?](https://www.kaggle.com/competitions/whats-cooking)
6. [Spooky Author Identification](https://www.kaggle.com/competitions/spooky-author-identification)
7. [LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)
8. [CommonLit Readability Prize](https://www.kaggle.com/competitions/commonlitreadabilityprize)
9. [Feedback Prize - Evaluating Student Writing](https://www.kaggle.com/competitions/feedback-prize-2021)
10. [Tweet Sentiment Extraction](https://www.kaggle.com/competitions/tweet-sentiment-extraction)

*Bonus* 

11. [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

## Data Assessment/Visualization: Poking and Proding
- There are no missing values

### Questions from Data Assessment
- If we run correlation feature selection how many features will be left due to the high correlation between features?
- Is data distribution density difference between train and original enough to affect the model?

## PROGRESS
