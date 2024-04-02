# Think Like a Data Scientist - Kaggle Playground 4.03 Solution
### What are the competition goals?
1. Given a set of well defined features, predict the age of abalone from various physical measurements. The dataset for this competition (both train and test) was generated from a deep learning model trained on the Abalone dataset. Feature distributions are close to, but not exactly the same, as the original. The predictions are evaluted using Root Mean Squared Log Error (RMSLE).
2. Produce this competition report detailing the step-by-step process of solving the competition goal above.
3. Learn and apply optuna ensembling method.
4. Rank in the top 10% of the competition's private leaderboard (3% better than last competition).

### What are my assumptions about the data and the competition?
*The aims of listing my assumptions is to check if they hold true and where not verifable acknowledge my assumptions*
1. The synthetic data has similar, if not exact, distribution as the original data.
2. 

### Has someone does this before?
Considering this is Kaggle competition, many brilliant minds have worked on various solutions. You never want to needlessly reinvent the wheel.

So browsing through the discussions and code section of the competition I have gained the following insights:

    

### Data Assessment: Poking and Proding
- There are no missing values
- The data distribution between train and test datasets are identical
- The data distribution density between train and original datasets is quite different. 
- The outliers distribution between train, test and original datasets are almost identical.
- All features are either float or int
- `id` column is an identifier and will most likely not be useful for prediction
- From histplots, it seems `Height`, `Shucked_Weight`, `Viscera_weight`, `Whole weight` and `Shell weight` have upper limit outliers. 
- The features with normal or normal-ish distributions are `Length` and `Diameter`
- `Sex` has 3 values (male, female and infant) so is the only categorical feature.
- All features are highly correlated with each with the lowest correlation being 0.9 between `Shucked weight` and `Height`. 
- `Shell weight` and `Height` are the most correlated with the `Rings` target.
- Correlation results are similar in original to train data but less values e.g. the lowest correlation features is still `Shucked weight` and `Height` but at 0.87.



### Questions from Data Assessment
- If we run correlation feature selection how many features will be left due to the high correlation between features?
- Is data distribution density difference between train and original enough to affect the model?

