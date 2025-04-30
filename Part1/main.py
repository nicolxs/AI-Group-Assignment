#encoding =utf-8
import sys
sys.stdout.reconfigure(encoding='utf-8')
from collections import defaultdict, deque
#import pandas
import pandas as pd
#read csv
scientists = pd.read_csv("Part1/csvs/scientists.csv")
#convert dataframe to dictionary
scientists_dict = scientists.to_dict(orient="records")

#read csv
authors = pd.read_csv("Part1/csvs/authors.csv")
#convert dataframe to dictionary
authors_dict = authors.to_dict(orient="records")

#read csv
papers = pd.read_csv("Part1/csvs/papers.csv")
#convert dataframe to dictionary
papers_dict = papers.to_dict(orient="records")


#scientist id to name mapping
scientist_names= {scientist["scientist_id"]: scientist["name"] for scientist in scientists_dict}

#graph for adjacency list
graph = defaultdict(list)

#group authors by paper
paper_authors = defaultdict(list)
for author in authors_dict:
    paper_authors[author["paper_id"]].append(author["scientist_id"])
    
#create connections between scientists who co-authored papers
for paper_id, authors_list in paper_authors.items():
    #for each pair of scientists who co-authored a paper, create a connection
    for i in range(len(authors_list)):
        for j in range(i + 1, len(authors_list)):
            scientist1 = authors_list[i]
            scientist2 = authors_list[j]
            graph[scientist1].append(scientist2)
            graph[scientist2].append(scientist1)
            
# BFS function to find degrees of separation
def find_degrees_of_separation(graph, start, end):
    if start == end:
        return 0
    
    visited = set()
    queue = deque([(start, 0)])  # (node, distance)
    visited.add(start)
    
    while queue:
        node, distance = queue.popleft()
        
        for neighbor in graph[node]:
            if neighbor == end:
                return distance + 1
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    
    return -1  # No path found

# Example usage
def test_degrees_of_separation():
    # Example to find degrees of separation between two scientists
    # Automate this part by selecting two random scientists from the graph
    # For demonstration, we will just take the first two scientists from the dictionary
    scientist1_id = list(scientist_names.keys())[0]  # First scientist 
    scientist2_id = list(scientist_names.keys())[1]  # Second scientist 
    
    degrees = find_degrees_of_separation(graph, scientist1_id, scientist2_id)
    
    if degrees == -1:
        print(f"No connection found between {scientist_names[scientist1_id]} and {scientist_names[scientist2_id]}")
    else:
        print(f"Degrees of separation between {scientist_names[scientist1_id]} and {scientist_names[scientist2_id]}: {degrees}")

# Print the graph structure (optional)
print("Graph structure (Scientist connections):")
for scientist_id, connections in graph.items():
    print(f"{scientist_names[scientist_id]} is connected to: {[scientist_names[conn] for conn in connections]}")
for scientist_id, connections in graph.items():
    print(f"{scientist_names[scientist_id]} is connected to: {[scientist_names[conn] for conn in connections]}")
for scientist_id, connections in graph.items():
    print(f"{scientist_names[scientist_id]} is connected to: {[scientist_names[conn] for conn in connections]}")

# Run the test
test_degrees_of_separation()


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


