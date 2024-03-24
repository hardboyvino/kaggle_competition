# Think Like a Data Scientist - Kaggle Playground 4.03 Solution

- Knowledge
- Technology
- Opinions

### What are the project goals?
1. Given a set of well defined features, predict the probability of various defects on steel plates. The features have been generated using deep learning model trained on an original dataset with feature distributions being close to, but not exactly the same, as the original. The predictions are evaluted using ROC AUC.
2. Produce a project report detailing the step-by-step process of solving the project goal above.
3. Learn and apply stacking ensembling method.

### What are my assumptions about the data and the competition?
* The aims of listing my assumptions is to check if they old true and where not verifable acknowledge my assumptions*
1. All steel plate have only 1 defect.
2. The synthetic data has similar, if not exact, distribution as the original data.
3. Generating more synthetic data using deep learning model will improve my results.

### Has someone does this before?
Considering this is Kaggle competition, many brilliant minds worked on various solutions. You never want to needlessly reinvent the wheel.

So browsing through the discussions and code section of the competition I have gained the following insights:
- There are 7 target classes in the original data with the steel plates havinG **ONLY** one type of fault, however in the synthetic data there are 11 different combinations of faults with some plates having more than 1 fault and some plates having no faults at all. This disproves the first assumption that every steel plate has only 1 defect.
    + This target class issue might be due to how the synthetic data was generated.
    + One can either drop the columns that do not confirm with my assumption or approach the competition as a multilabel instead of a multiclass.
    [Difference between Multiclass and Multilabel](https://www.geeksforgeeks.org/an-introduction-to-multilabel-classification/)
    + Add an 8th class of 'No Fault' or add them to 'Other_Faults' for the observation with no fault. This is because EDA shows that the steel plates most likely have faults and have been mislabelled.
    [Source](https://www.kaggle.com/competitions/playground-series-s4e3/discussion/480805)

- A detailed explanations of what each feature means [Explanation 1.](https://www.kaggle.com/competitions/playground-series-s4e3/discussion/480936) [Explanation 2.](https://www.kaggle.com/competitions/playground-series-s4e3/discussion/481006)
    + Pastry sounds like the opposite of Bumps. Pastry is like a dent while Bumps is a raised area on the steel plate.
    + Some faults might have more luminos characteristics than others.

- Using NN as model [Discussion](https://www.kaggle.com/competitions/playground-series-s4e3/discussion/481167)

- Got some feature engineering ideas.
    + [Idea bank 1](https://www.kaggle.com/competitions/playground-series-s4e3/discussion/481687)
    + [Idea bank 2](https://www.kaggle.com/competitions/playground-series-s4e3/discussion/482475)
    



    ### TO-DO
    - Observe the relationship between each fault and the features individually
    - Data EDA and preprocessing following this [notebook](https://www.kaggle.com/code/mahsateimourikia/steel-plates-fault-analysis)