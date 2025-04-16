import sys
sys.stdout.reconfigure(encoding='utf-8')
#import pandas
import pandas as pd
#read csv
scientists = pd.read_csv("Part1/csvs/scientists.csv")
#convert dataframe to dictionary
scientists_dict = scientists.to_dict(orient="records")
print(scientists_dict)
#read csv
authors = pd.read_csv("Part1/csvs/authors.csv")
#convert dataframe to dictionary
authors_dict = authors.to_dict(orient="records")
print(authors_dict)
#read csv
papers = pd.read_csv("Part1/csvs/papers.csv")
#convert dataframe to dictionary
papers_dict = papers.to_dict(orient="records")
print(papers_dict)


# from tokenize import String


# scientists = {
#     "scientist_id": int,
#     "name": String,
#     }

# papers = {
#     "paper_id": int,
#     "title": String,
#     "year": int,
# }
    
# authors = {
#     "paper_id": int,
#     "scientist_id": int,
# }


