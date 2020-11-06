# Data 401 Project 2
## Team Mirimax: Michal Golovanevsky, Jenna Landy, Andrew Sulistio

This project is two fold. Looking at a dataset of movie reviews, we first aim to develop a feature set that performs better than the bag of words provided to us. Second, we implement three linear classifiers and compare their performances on (1) our feature set, (2) the provided bag of words, and (3) how the accuracies of the classifiers compare between the datasets. Our final classifiers performed at about 70%-75% accuracy, compared to around 84% accuracy for the given features. However, our features reduced dimensionality significantly, with our dimensions being 0.01%  the size of the original features. LDA and Logistic Regression performed very similarly, but LDA has the advantage of having a closed form solution. 

*Key Words* logistic regression, linear discriminant analysis, support vector machine, IMDb movie reviews

**Directory Structure**
- opinion_lexicon: directory of files from opinion lexicon https://www.kaggle.com/nltkdata/opinion-lexicon/download#README.txt
- project2_data: directory of data files saved in Project2.ipynb
- project2.py: python script with classifier implementations and constants
- Project2.ipynb: full workflow of this project
- Feature Extraction.ipynb: calculations of polarity scores
- Visualizations.ipynb: creating confusion matrices and tables

**Variables**

Unique Identifier:
- filename: full path to the file

Predictors:
- number of sentences
- average number of words per sentence
- average word length
- dale chall readability score
    - calculated using the py-readability-metrics package (https://pypi.org/project/py-readability-metrics/)
- positive emoticons: number of positive emoticons adjusted for number of words in the review
- negative emoticons: number of negative emoticons adjusted for number of words in the review
    - positive and negative emoticons determined from https://en.wikipedia.org/wiki/List_of_emoticons
- positive words: number of positive words (or negative words with a negator in the sentence), weighted by looking at boosting and diminishing words in the same sentence and adjusted for number of words in the review
- negative words: number of negative words (or positive words with a negator in the sentence), weighted by looking at boosting and diminishing words in the same sentence and adjusted for number of words in the review
    - positive and negative words determined from the Opinion Lexicon made available by University of Illinois at Chicago and downloadable from kaggle https://www.kaggle.com/nltkdata/opinion-lexicon/download#README.txt
- polarity:
    - see page 2 of the report


Response:
- sentiment: 1 for positive, 0 for negative
