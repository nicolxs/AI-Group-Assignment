import pandas as pd
from collections import deque
import sys

# Load the data from the CSV files
def load_data(authors_file, papers_file, scientists_file):
    """Loads data from CSV files and creates necessary dictionaries."""
    try:
        df_authors = pd.read_csv(authors_file, dtype={'paper_id': str, 'scientist_id': str})
        df_papers = pd.read_csv(papers_file, dtype={'paper_id': str})
        df_scientists = pd.read_csv(scientists_file, dtype={'scientist_id': str})

        scientist_name_to_id = dict(zip(df_scientists['name'], df_scientists['scientist_id']))

        scientist_id_to_info = {}
        for scientist_id in df_scientists['scientist_id']:
            name = df_scientists.loc[df_scientists['scientist_id'] == scientist_id, 'name'].iloc[0]
            papers = set(df_authors.loc[df_authors['scientist_id'] == scientist_id, 'paper_id'])
            scientist_id_to_info[scientist_id] = {'name': name, 'papers': papers}

        paper_id_to_info = {}
        for paper_id in df_papers['paper_id']:
            title = df_papers.loc[df_papers['paper_id'] == paper_id, 'title'].iloc[0]
            year = df_papers.loc[df_papers['paper_id'] == paper_id, 'year'].iloc[0]
            authors = set(df_authors.loc[df_authors['paper_id'] == paper_id, 'scientist_id'])
            # Ensure authors set is not empty before adding to paper_id_to_info
            if authors:
                paper_id_to_info[paper_id] = {'title': title, 'year': year, 'authors': authors}

        # Create the co-authorship graph
        coauthor_graph = {}
        for paper_id, paper_info in paper_id_to_info.items():
            authors = paper_info['authors']
            for author in authors:
                if author not in coauthor_graph:
                    coauthor_graph[author] = set()
                for coauthor in authors:
                    if author != coauthor:
                        coauthor_graph[author].add((paper_id, coauthor))

        return scientist_name_to_id, scientist_id_to_info, paper_id_to_info, coauthor_graph

    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure the CSV files are in the correct directory.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred during data loading and processing: {e}")
        return None, None, None, None

# Implement the shortest_path function using Breadth-First Search (BFS)
def shortest_path(coauthor_graph, source, target):
    """
    Finds the shortest path between two scientists in a co-authorship graph using BFS.

    Args:
        coauthor_graph: A dictionary representing the co-authorship network as an adjacency list.
        source: The scientist ID of the source scientist.
        target: The scientist ID of the target scientist.

    Returns:
        A list of (paper_id, scientist_id) tuples representing the shortest path, or None if no path exists.
    """
    if source not in coauthor_graph or target not in coauthor_graph:
        return None

    queue = deque([(source, [])])  # Queue of (scientist_id, path)
    visited = set()
    visited.add(source) # Mark the source node as visited

    while queue:
        current_scientist, path = queue.popleft()

        if current_scientist == target:
            return path

        if current_scientist in coauthor_graph:
            for paper_id, next_scientist in coauthor_graph[current_scientist]:
                if next_scientist not in visited:
                    visited.add(next_scientist)
                    new_path = path + [(paper_id, next_scientist)]
                    queue.append((next_scientist, new_path))

    return None

# Main part of the program
if __name__ == "__main__":
    if len(sys.argv) != 2:
        # Modified to work without command line arguments initially, as the image shows interactive input
        # print("Usage: python your_program_name.py <data_directory>")
        # sys.exit(1)
        pass # Allow execution without arguments for interactive input

    # data_directory = sys.argv[1] # Not used in the interactive version

    print("Loading data...")
    # Assuming the CSV files are in the same directory as the script
    authors_file = 'authors.csv'
    papers_file = 'papers.csv'
    scientists_file = 'scientists.csv'

    scientist_name_to_id, scientist_id_to_info, paper_id_to_info, coauthor_graph = load_data(authors_file, papers_file, scientists_file)

    if not all([scientist_name_to_id, scientist_id_to_info, paper_id_to_info, coauthor_graph]):
        sys.exit(1)

    print("Data loaded.")

    # Get scientist names from user input
    source_name = input("Name: ")
    target_name = input("Name: ")

    # Get scientist IDs from names
    source_id = scientist_name_to_id.get(source_name)
    target_id = scientist_name_to_id.get(target_name)

    if not source_id:
        print(f"Scientist '{source_name}' not found in the data.")
        sys.exit(1)

    if not target_id:
        print(f"Scientist '{target_name}' not found in the data.")
        sys.exit(1)

    # Find the shortest path
    path = shortest_path(coauthor_graph, source_id, target_id)

    if path:
        degrees_of_separation = len(path)
        print(f"{degrees_of_separation} degrees of separation.")

        # Print the path details
        current_scientist_id = source_id
        for i, (paper_id, next_scientist_id) in enumerate(path):
            paper_title = paper_id_to_info[paper_id]['title']
            current_scientist_name = scientist_id_to_info[current_scientist_id]['name']
            next_scientist_name = scientist_id_to_info[next_scientist_id]['name']
            print(f"{i + 1}: {current_scientist_name} and {next_scientist_name} co-authored \"{paper_title}\"")
            current_scientist_id = next_scientist_id # Move to the next scientist in the path
    else:
        print(f"No path found between {source_name} and {target_name}. They are not connected in the co-authorship network.")