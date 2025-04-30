import copy
from collections import deque

class Sudoku_AI_solver:
    def __init__(self, board):
        """Initialize the Sudoku solver with a board."""
        self.board = board
        self.domains = {}
        self.initialize_domains()
        
    def initialize_domains(self):
        """Initialize the domains for each cell in the Sudoku board."""
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    self.domains[(i, j)] = set(range(1, 10))
                else:
                    self.domains[(i, j)] = {self.board[i][j]}
    
    def enforce_node_consistency(self):
        """Remove any values from a cell's domain that violate the Sudoku rules."""
        for cell in self.domains:
            i, j = cell
            if len(self.domains[cell]) == 1:
                continue  
                
           
            peers = self.get_peers(cell)
            assigned_peers = {self.board[x][y] for (x, y) in peers if self.board[x][y] != 0}
            
            
            self.domains[cell] -= assigned_peers
    
    def get_peers(self, cell):
        """Return all cells that are in the same row, column, or 3x3 subgrid."""
        i, j = cell
        peers = set()
        
       
        peers.update((i, y) for y in range(9) if y != j)
        
     
        peers.update((x, j) for x in range(9) if x != i)
        
      
        subgrid_i = (i // 3) * 3
        subgrid_j = (j // 3) * 3
        peers.update((x, y) for x in range(subgrid_i, subgrid_i + 3) 
                          for y in range(subgrid_j, subgrid_j + 3) 
                          if (x, y) != (i, j))
        
        return peers
    
    def revise(self, x, y):
        """Make variable x arc consistent with variable y."""
        revised = False
        x_domain = set(self.domains[x])
        y_domain = set(self.domains[y])
        
        
        if len(y_domain) == 1:
            value = next(iter(y_domain))
            if value in x_domain and len(x_domain) > 1:
                self.domains[x].remove(value)
                revised = True
        else:
            
            if len(x_domain) == 1:
                value = next(iter(x_domain))
                if value in y_domain and len(y_domain) > 1:
                    self.domains[y].remove(value)
                    revised = True
        
        return revised
    
    def ac3(self):
        """Use the AC-3 algorithm to make the entire puzzle arc consistent."""
        queue = deque()
        
       
        for cell in self.domains:
            peers = self.get_peers(cell)
            for peer in peers:
                queue.append((cell, peer))
        
        while queue:
            x, y = queue.popleft()
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False  
                
                
                peers = self.get_peers(x)
                for peer in peers:
                    if peer != y:
                        queue.append((peer, x))
        
        return True
    
    def assignment_complete(self, assignment):
        """Return True if every cell has been assigned a value (no 0s remain)."""
        for i in range(9):
            for j in range(9):
                if assignment[i][j] == 0:
                    return False
        return True
    
    def consistent(self, assignment):
        """Check if the current assignment does not violate any Sudoku constraints."""
        
        for i in range(9):
            row = [assignment[i][j] for j in range(9) if assignment[i][j] != 0]
            if len(row) != len(set(row)):
                return False
        
        for j in range(9):
            col = [assignment[i][j] for i in range(9) if assignment[i][j] != 0]
            if len(col) != len(set(col)):
                return False
        
        
        for subgrid_i in range(0, 9, 3):
            for subgrid_j in range(0, 9, 3):
                subgrid = []
                for i in range(subgrid_i, subgrid_i + 3):
                    for j in range(subgrid_j, subgrid_j + 3):
                        if assignment[i][j] != 0:
                            subgrid.append(assignment[i][j])
                if len(subgrid) != len(set(subgrid)):
                    return False
        
        return True
    
    def order_domain_values(self, var, assignment):
        """Return values in the domain of var, ordered by least constraining value."""
        i, j = var
        value_counts = []
        
        for value in self.domains[var]:
            count = 0
            
            peers = self.get_peers(var)
            for (x, y) in peers:
                if assignment[x][y] == 0 and value in self.domains[(x, y)]:
                    count += 1
            value_counts.append((count, value))
        
       
        value_counts.sort()
        return [value for (count, value) in value_counts]
    
    def select_unassigned_variable(self, assignment):
        """Use the MRV and Degree heuristics to choose the next variable to assign."""
        unassigned = []
        
        for i in range(9):
            for j in range(9):
                if assignment[i][j] == 0:
                    domain_size = len(self.domains[(i, j)])
                    
                    peers = self.get_peers((i, j))
                    degree = sum(1 for (x, y) in peers if assignment[x][y] == 0)
                    unassigned.append((domain_size, degree, (i, j)))
        
        
        unassigned.sort()
        return unassigned[0][2] if unassigned else None
    
    def backtrack(self, assignment):
        """Recursively try to solve the puzzle using a backtracking search."""
        if self.assignment_complete(assignment):
            return assignment
        
        var = self.select_unassigned_variable(assignment)
        i, j = var
        
        for value in self.order_domain_values(var, assignment):
            new_assignment = copy.deepcopy(assignment)
            new_assignment[i][j] = value
            
            if self.consistent(new_assignment):
               
                new_solver = Sudoku_AI_solver(new_assignment)
                new_solver.domains = copy.deepcopy(self.domains)
                new_solver.domains[var] = {value}
                
                
                if new_solver.ac3():
                    result = new_solver.backtrack(new_assignment)
                    if result is not None:
                        return result
        
        return None
    
    def solve(self):
        """Solve the Sudoku puzzle and return the solution."""
     
        self.enforce_node_consistency()
        
       
        if not self.ac3():
            return None  
        
        solution = self.backtrack(self.board)
        return solution
    
    @staticmethod
    def print_board(board):
        """Helper function to print the Sudoku board."""
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("-" * 21)
            row = []
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    row.append("|")
                row.append(str(board[i][j]) if board[i][j] != 0 else " ")
            print(" ".join(row))


# Example:
if __name__ == "__main__":
    # Example Sudoku board (0 represents empty cells)
    example_board = [
        [0 ,0, 0, 6, 0, 0, 4, 0, 0],
        [7, 0, 0, 0, 0, 3, 6, 0, 0],
        [0, 0, 0, 0, 9, 1, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 0, 1, 8, 0, 0, 0, 3],
        [0, 0, 0, 3, 0, 6, 0, 4, 5],
        [0, 4, 0, 2, 0, 0, 0, 6, 0],
        [9, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 1, 0, 0],
    ]


    
    solver = Sudoku_AI_solver(example_board)
    solution = solver.solve()
    
    if solution:
        print("\nSolved Sudoku:")
        Sudoku_AI_solver.print_board(solution)
    else:
        print("\nNo solution exists for this Sudoku.")