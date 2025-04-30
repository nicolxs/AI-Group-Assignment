# encoding=utf-8
import sys
sys.stdout.reconfigure(encoding='utf-8')
from collections import defaultdict, deque
import pandas as pd
import os

# Global variables to store data
people = {}  # Maps scientist_id to scientist details
papers = {}  # Maps paper_id to paper details
names = defaultdict(set)  # Maps lowercase names to sets of scientist IDs
neighbors = defaultdict(set)  # Maps scientist_id to set of (paper_id, scientist_id) neighbors

def load_data(directory):
    """
    Load data from CSV files in the specified directory.
    """
    # Load scientists
    scientists = pd.read_csv(os.path.join(directory, "scientists.csv"))
    for _, scientist in scientists.iterrows():
        people[str(scientist["scientist_id"])] = {
            "name": scientist["name"],
            "scientist_id": str(scientist["scientist_id"])
        }
        # Add name to lowercase mapping for case-insensitive lookup
        names[scientist["name"].lower()].add(str(scientist["scientist_id"]))

    # Load papers
    papers_df = pd.read_csv(os.path.join(directory, "papers.csv"))
    for _, paper in papers_df.iterrows():
        papers[str(paper["paper_id"])] = {
            "title": paper["title"],
            "paper_id": str(paper["paper_id"])
        }

    # Load authors and build the neighbor connections
    authors = pd.read_csv(os.path.join(directory, "authors.csv"))
    
    # First, create a mapping from paper_id to scientist_ids
    paper_to_scientists = defaultdict(set)
    for _, author in authors.iterrows():
        paper_id = str(author["paper_id"])
        scientist_id = str(author["scientist_id"])
        paper_to_scientists[paper_id].add(scientist_id)
    
    # Then, for each paper, create connections between all co-authors
    for paper_id, scientists in paper_to_scientists.items():
        for scientist1 in scientists:
            for scientist2 in scientists:
                if scientist1 != scientist2:
                    neighbors[scientist1].add((paper_id, scientist2))

def shortest_path(source, target):
    """
    Returns the shortest path from source scientist to target scientist as a list
    of (paper_id, scientist_id) pairs. Returns None if no path exists.
    """
    # Check if source and target are the same
    if source == target:
        return []
    
    # Initialize the frontier with the source and an empty path
    frontier = deque()
    frontier.append((source, []))
    
    # Keep track of explored scientists to avoid cycles
    explored = set()
    
    while frontier:
        # Get the current scientist and the path to reach them
        current, path = frontier.popleft()
        
        # Mark as explored
        explored.add(current)
        
        # Check all neighbors of the current scientist
        for paper_id, neighbor in neighbors[current]:
            # Skip already explored scientists
            if neighbor not in explored:
                # Create the new path by extending the current path
                new_path = path + [(paper_id, neighbor)]
                
                # If we've found the target, return the path
                if neighbor == target:
                    return new_path
                
                # Otherwise, add this neighbor to the frontier for further exploration
                frontier.append((neighbor, new_path))
    
    # If we've exhausted the frontier and haven't found a path, return None
    return None

def analyze_entire_network():
    """
    Analyzes the entire network by finding connections between all scientists.
    Generates statistics and finds interesting connections without user input.
    """
    print("\n=== Analyzing Entire Scientist Collaboration Network ===\n")
    
    # Get all scientist IDs
    all_scientists = list(people.keys())
    total_scientists = len(all_scientists)
    
    print(f"Total scientists in dataset: {total_scientists}")
    
    # Statistics tracking
    connection_counts = {}
    max_separation = 0
    max_path = None
    max_path_scientists = (None, None)
    disconnected_pairs = 0
    total_pairs = 0
    
    # Sample size - use a smaller number for faster testing
    # Change this to total_scientists for complete analysis (may be slow)
    sample_size = min(100, total_scientists)
    
    print(f"Analyzing connections among {sample_size} scientists (sampling for efficiency)...")
    
    # Limit the analysis to a sample of scientists
    sample_scientists = all_scientists[:sample_size]
    
    # Process all pairs of scientists in the sample
    for i, source_id in enumerate(sample_scientists):
        if i % 10 == 0 and i > 0:
            print(f"Processed {i}/{sample_size} scientists...")
        
        for target_id in sample_scientists[i+1:]:
            total_pairs += 1
            path = shortest_path(source_id, target_id)
            
            if path is None:
                disconnected_pairs += 1
            else:
                path_length = len(path)
                
                # Update statistics
                connection_counts[path_length] = connection_counts.get(path_length, 0) + 1
                
                # Track maximum separation
                if path_length > max_separation:
                    max_separation = path_length
                    max_path = path
                    max_path_scientists = (source_id, target_id)
    
    # Calculate percentage of connected scientists
    connected_percentage = (total_pairs - disconnected_pairs) / total_pairs * 100 if total_pairs > 0 else 0
    
    # Report results
    print("\n=== Network Analysis Results ===\n")
    print(f"Connected pairs: {total_pairs - disconnected_pairs}/{total_pairs} ({connected_percentage:.1f}%)")
    print("\nDistribution of degrees of separation:")
    
    for separation, count in sorted(connection_counts.items()):
        percentage = count / (total_pairs - disconnected_pairs) * 100 if (total_pairs - disconnected_pairs) > 0 else 0
        print(f"  {separation} degrees: {count} pairs ({percentage:.1f}%)")
    
    # Display the longest path if one was found
    if max_path:
        source_id, target_id = max_path_scientists
        print(f"\nMaximum separation found: {max_separation} degrees")
        print(f"Longest path: {people[source_id]['name']} to {people[target_id]['name']}")
        print("\nPath details:")
        
        current = source_id
        for i, (paper_id, next_scientist) in enumerate(max_path):
            paper = papers[paper_id]["title"]
            scientist1 = people[current]["name"]
            scientist2 = people[next_scientist]["name"]
            print(f"{i + 1}: {scientist1} and {scientist2} co-authored \"{paper}\"")
            current = next_scientist
    
    print("\n=== End of Degrees of Seperation Analysis ===\n")
1
def main():
    if len(sys.argv) != 2:
        # Default to "Part1/csvs" if no directory is specified
        data_dir = os.path.join(os.path.dirname(__file__), "csvs")
        
        print(f"No data directory specified. Using default: {data_dir}")
    else:
        data_dir = sys.argv[1]
        print(f"Using specified data directory: {data_dir}")

    
    try:
        load_data(data_dir)
        print("Data loaded successfully.")
    except Exception as e:
        sys.exit(f"Error loading data from {data_dir}: {e}")
    
    while True:
        print("\nOptions:")
        print("1. Look up connection between two scientists")
        print("2. Analyze entire collaboration network")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "3":
            break
        elif choice == "2":
            analyze_entire_network()
        elif choice == "1":
            try:
                # Get input from user
                name1 = input("First scientist: ")
                name2 = input("Second scientist: ")
                
                # Look up scientist IDs
                ids1 = list(names.get(name1.lower(), set()))
                ids2 = list(names.get(name2.lower(), set()))
                
                # Handle cases where names aren't found
                if not ids1:
                    print(f"Scientist '{name1}' not found. Try again.")
                    continue
                if not ids2:
                    print(f"Scientist '{name2}' not found. Try again.")
                    continue
                
                # If there are multiple scientists with the same name, let the user choose
                if len(ids1) > 1:
                    print(f"Multiple scientists named '{name1}' found:")
                    for i, scientist_id in enumerate(ids1):
                        print(f"{i+1}. {people[scientist_id]['name']}")
                    selection = int(input("Enter the number of your selection: ")) - 1
                    source = ids1[selection]
                else:
                    source = ids1[0]
                    
                if len(ids2) > 1:
                    print(f"Multiple scientists named '{name2}' found:")
                    for i, scientist_id in enumerate(ids2):
                        print(f"{i+1}. {people[scientist_id]['name']}")
                    selection = int(input("Enter the number of your selection: ")) - 1
                    target = ids2[selection]
                else:
                    target = ids2[0]
                
                # Find and display the path
                print(f"\nFinding connection from {people[source]['name']} to {people[target]['name']}...")
                path = shortest_path(source, target)
                
                if path is None:
                    print("No connection found.")
                else:
                    print(f"{len(path)} degrees of separation.")
                    current = source
                    for i, (paper_id, next_scientist) in enumerate(path):
                        paper = papers[paper_id]["title"]
                        scientist1 = people[current]["name"]
                        scientist2 = people[next_scientist]["name"]
                        print(f"{i + 1}: {scientist1} and {scientist2} co-authored \"{paper}\"")
                        current = next_scientist
                        
                print("\n" + "-" * 80)
                
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Let's try again.")
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
