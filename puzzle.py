import time # Optional: for timing the solver
import copy # Needed for deep copying domains during inference in backtracking (optional but safer)

class Sudoku_AI_solver:
    """
    A class to solve Sudoku puzzles using Constraint Satisfaction Problem (CSP)
    techniques, specifically Arc Consistency (AC-3) and backtracking search
    with Minimum Remaining Values (MRV) heuristic.
    """
    def __init__(self, board):
        """
        Initializes the Sudoku solver.

        Args:
            board (list[list[int]]): A 9x9 list representing the Sudoku board.
                                     0 represents an empty cell.
        """
        if len(board) != 9 or any(len(row) != 9 for row in board):
            raise ValueError("Board must be 9x9")
        self.board = board
        # Domains store the possible values for each cell (row, col)
        self.domains = self._initialize_domains()
        # Store neighbors for quick lookup during AC-3 and consistency checks
        self.neighbors = self._calculate_all_neighbors()

    def _initialize_domains(self):
        """
        Initializes the domains for each cell based on the initial board.
        Empty cells get {1, 2, ..., 9}. Filled cells get {value}.
        """
        domains = {}
        for r in range(9):
            for c in range(9):
                cell = (r, c)
                if self.board[r][c] == 0:
                    # Possible values for an empty cell
                    domains[cell] = set(range(1, 10))
                else:
                    # Fixed value for a pre-filled cell
                    value = self.board[r][c]
                    if not (1 <= value <= 9):
                         raise ValueError(f"Invalid number {value} at ({r},{c}) in input board.")
                    domains[cell] = {value}
        return domains

    def _get_neighbors(self, var):
        """
        Helper function to get all unique neighboring cells (same row, column, or 3x3 box)
        for a given cell. Neighbors are cells that share a constraint.

        Args:
            var (tuple): The cell coordinates (row, col).

        Returns:
            set: A set of neighbor cell coordinates (tuples).
        """
        r, c = var
        neighbors = set()
        # Row & Col neighbors (excluding the cell itself)
        for i in range(9):
            if i != c: neighbors.add((r, i)) # Row
            if i != r: neighbors.add((i, c)) # Col
        # 3x3 Box neighbors (excluding the cell itself)
        start_row, start_col = 3 * (r // 3), 3 * (c // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if (i, j) != var:
                    neighbors.add((i, j))
        return neighbors

    def _calculate_all_neighbors(self):
        """Pre-calculates and stores neighbors for all cells for efficiency."""
        all_neighbors = {}
        for r in range(9):
            for c in range(9):
                all_neighbors[(r, c)] = self._get_neighbors((r, c))
        return all_neighbors

    def enforce_node_consistency(self):
        """
        Ensures initial node consistency. For Sudoku, this means checking if any
        pre-filled cell's value violates the basic constraints with other pre-filled cells.
        This is largely covered by the initial consistency check within solve(),
        but can be called explicitly.
        """
        # Check if any pre-filled cell conflicts with another pre-filled cell
        initial_assignment = {var: list(domain)[0] for var, domain in self.domains.items() if len(domain) == 1}
        if not self.consistent(initial_assignment):
            print("Warning: Initial board configuration is inconsistent.")
            return False
        return True

    def revise(self, x, y):
        """
        Makes variable `x` arc-consistent with respect to variable `y`.
        Removes values from domain(x) if they have no possible corresponding value in domain(y)
        that satisfies the constraint (value_x != value_y). This is the core check for Sudoku constraints.

        Args:
            x (tuple): Coordinates of the first cell (variable).
            y (tuple): Coordinates of the second cell (variable, must be a neighbor of x).

        Returns:
            bool: True if the domain of x was revised (reduced), False otherwise.
        """
        revised = False
        domain_x = list(self.domains[x]) # Iterate over a copy

        for val_x in domain_x:
            # Check if there exists *any* value in domain(y) such that val_x != val_y
            # If y only has one value, and it's the same as val_x, then val_x is impossible for x.
            if len(self.domains[y]) == 1 and val_x in self.domains[y]:
                 self.domains[x].remove(val_x)
                 revised = True

        return revised

    def ac3(self, initial_queue=None):
        """
        Applies the AC-3 algorithm to enforce arc consistency across the Sudoku grid.
        Reduces the domains of variables based on constraints with their neighbors.

        Args:
            initial_queue (list, optional): An initial queue of arcs (tuples of variables)
                                             to process. If None, initializes with all arcs.

        Returns:
            bool: True if arc consistency is achieved without emptying any domain,
                  False if an inconsistency is found (a domain becomes empty).
        """
        if initial_queue is None:
            # Initialize queue with all arcs (constraints between neighboring cells)
            queue = []
            all_vars = list(self.domains.keys())
            for var in all_vars:
                for neighbor in self.neighbors[var]:
                    # Add arc (var -> neighbor) and (neighbor -> var) if not already added implicitly
                    # Standard AC3 considers directed arcs x -> y
                    queue.append((var, neighbor))
        else:
            queue = initial_queue

        while queue:
            x, y = queue.pop(0) # Use pop(0) for FIFO queue behavior

            # Try to revise the domain of x based on y
            if self.revise(x, y):
                # If domain(x) becomes empty after revision, the puzzle is inconsistent
                if not self.domains[x]:
                    return False # Inconsistency found

                # If domain(x) was reduced, we need to re-check arcs pointing TO x
                # Add arcs (z, x) for all neighbors z of x (excluding y itself, as per standard AC3)
                for z in self.neighbors[x]:
                    if z != y: # Standard AC3 optimization
                       queue.append((z, x))

        return True # Arc consistency achieved for the given arcs

    def assignment_complete(self, assignment):
        """
        Checks if the current assignment covers all variables (all 81 cells).

        Args:
            assignment (dict): A dictionary mapping cell coordinates (tuple) to assigned values (int).

        Returns:
            bool: True if all 81 cells have been assigned a value, False otherwise.
        """
        return len(assignment) == 81

    def consistent(self, assignment):
        """
        Checks if the current assignment (potentially partial) is consistent
        with Sudoku rules (no duplicate values in rows, columns, or 3x3 boxes
        among the assigned cells).

        Args:
            assignment (dict): A dictionary mapping cell coordinates (tuple) to assigned values (int).

        Returns:
            bool: True if the assignment is consistent, False otherwise.
        """
        assigned_vars = list(assignment.keys())
        # Check for conflicts between every pair of assigned variables
        for i in range(len(assigned_vars)):
            var1 = assigned_vars[i]
            val1 = assignment[var1]

            # Check against all other assigned variables that are neighbors
            for neighbor in self.neighbors[var1]:
                if neighbor in assignment and assignment[neighbor] == val1:
                    return False # Found a conflict with a neighbor

        return True # No conflicts found

    def order_domain_values(self, var, assignment):
        """
        Orders the values in the domain of a variable for the backtracking search.
        Currently uses simple numerical order. LCV (Least Constraining Value)
        could be implemented here for potentially better performance.

        Args:
            var (tuple): The variable (cell coordinate) whose domain values are to be ordered.
            assignment (dict): The current assignment (needed for LCV).

        Returns:
            list: An ordered list of values from the domain of 'var'.
        """
        # Simple approach: return values in numerical order
        return sorted(list(self.domains[var]))

        # --- LCV Implementation Sketch (More complex, uncomment to try) ---
        # def count_conflicts(value_to_try):
        #     """Counts how many choices this value eliminates for neighbors."""
        #     count = 0
        #     for neighbor in self.neighbors[var]:
        #         if neighbor not in assignment and value_to_try in self.domains[neighbor]:
        #             count += 1
        #     return count
        # # Return values sorted by the number of conflicts they cause (ascending)
        # return sorted(list(self.domains[var]), key=count_conflicts)
        # --- End LCV Sketch ---


    def select_unassigned_variable(self, assignment):
        """
        Selects the next unassigned variable to try assigning a value to.
        Uses the Minimum Remaining Values (MRV) heuristic: chooses the variable
        with the smallest current domain size (fewest possible values).
        Ties are broken by choosing the first one encountered in this simple implementation.
        (Could add Degree Heuristic as a tie-breaker).

        Args:
            assignment (dict): The current assignment.

        Returns:
            tuple or None: The coordinates (row, col) of the selected variable,
                           or None if all variables are assigned.
        """
        unassigned_vars = [var for var in self.domains if var not in assignment]
        if not unassigned_vars:
            return None # All variables assigned

        # MRV heuristic: Find the variable with the smallest domain size
        min_domain_size = float('inf')
        selected_var = None
        for var in unassigned_vars:
            domain_size = len(self.domains[var])
            if domain_size < min_domain_size:
                min_domain_size = domain_size
                selected_var = var
                # Optimization: If domain size is 1, it's the best we can do
                if min_domain_size == 1:
                    break

        # Optional: Degree Heuristic as tie-breaker (if multiple vars have the same min_domain_size)
        # This can sometimes help but adds complexity. MRV alone is often effective.
        # tied_vars = [v for v in unassigned_vars if len(self.domains[v]) == min_domain_size]
        # if len(tied_vars) > 1:
        #    max_degree = -1
        #    for v in tied_vars:
        #        # Count unassigned neighbors
        #        degree = sum(1 for neighbor in self.neighbors[v] if neighbor not in assignment)
        #        if degree > max_degree:
        #            max_degree = degree
        #            selected_var = v # Update selected_var based on degree

        return selected_var


    def backtrack(self, assignment):
        """
        Performs recursive backtracking search to find a valid Sudoku solution.

        Args:
            assignment (dict): The current state of the assignment (cell -> value).
                               This dictionary is modified during the search.

        Returns:
            dict or None: A complete and valid assignment (solution) if found, otherwise None.
        """
        # Base Case: If assignment is complete, we found a solution
        if self.assignment_complete(assignment):
            # Final consistency check (should be consistent if logic is correct, but good practice)
            if self.consistent(assignment):
                return assignment
            else:
                # This case should ideally not be reached if consistent() is checked before recursion
                print("Error: Reached complete assignment that is inconsistent.")
                return None

        # Select the next variable to assign using MRV heuristic
        var = self.select_unassigned_variable(assignment)
        if var is None:
             # Should be caught by assignment_complete, but safety check
             return assignment if self.assignment_complete(assignment) else None

        # Try assigning values from the domain in the specified order
        original_domain = self.domains[var].copy() # Keep track if needed for inference restoration

        for value in self.order_domain_values(var, assignment):
            # 1. Try assigning the value
            assignment[var] = value

            # 2. Check consistency of the new partial assignment
            if self.consistent(assignment):
                # --- Optional: Inference (Forward Checking or Maintaining Arc Consistency) ---
                # Inference can significantly speed up the search by pruning domains
                # of neighbors after assigning a value. AC-3 is stronger.
                # We need to store the original domains to restore them if this path fails.
                # Simple Forward Checking Example:
                # removed_values = {} # Store {neighbor: set_of_removed_values}
                # consistent_after_inference = True
                # for neighbor in self.neighbors[var]:
                #     if neighbor not in assignment:
                #         if value in self.domains[neighbor]:
                #             self.domains[neighbor].remove(value)
                #             if neighbor not in removed_values: removed_values[neighbor] = set()
                #             removed_values[neighbor].add(value)
                #             if not self.domains[neighbor]: # Domain wiped out by forward check
                #                 consistent_after_inference = False
                #                 break
                # if consistent_after_inference:
                #     result = self.backtrack(assignment) # Recurse
                #     if result is not None: return result # Solution found!
                #
                # # Restore domains changed by inference before backtracking
                # for neighbor, removed_set in removed_values.items():
                #     self.domains[neighbor].update(removed_set)
                # --- End Optional Inference Section ---

                # --- Backtracking without Inference ---
                result = self.backtrack(assignment.copy()) # Pass a copy if not doing inference restoration
                # result = self.backtrack(assignment) # Pass original if doing inference restoration
                if result is not None:
                    return result # Solution found, propagate it back up
                # --- End Backtracking without Inference ---


            # 3. If assignment was inconsistent or the recursive call returned None (dead end),
            #    backtrack: remove the current assignment for 'var' and try the next value.
            del assignment[var] # Backtrack: Remove var assignment

        # If no value for 'var' worked in the loop, this path is a dead end for the current assignment state
        return None


    def solve(self):
        """
        Solves the Sudoku puzzle using AC-3 preprocessing followed by backtracking search.

        Returns:
            list[list[int]] or None: The solved 9x9 Sudoku board if a solution is found,
                                      otherwise None if the puzzle is unsolvable.
        """
        print("Starting solver...")
        start_time = time.time() # Optional timing

        # 1. Enforce initial node consistency (basic check on pre-filled cells)
        if not self.enforce_node_consistency():
             print("Initial board is inconsistent based on pre-filled numbers.")
             return None

        # 2. Enforce Arc Consistency (AC-3) to prune domains initially
        print("Running AC-3 preprocessing...")
        if not self.ac3():
            print("AC-3 determined the puzzle is unsolvable.")
            return None # AC-3 found an inconsistency

        # 3. Prepare initial assignment for backtracking
        # Include cells that are now uniquely determined (domain size is 1 after AC-3)
        initial_assignment = {}
        print("Building initial assignment from determined cells...")
        for var, domain in self.domains.items():
            if len(domain) == 1:
                initial_assignment[var] = list(domain)[0]
            elif not domain: # Should have been caught by ac3, but double-check
                print(f"Error: Empty domain found for cell {var} after AC-3.")
                return None

        # Check consistency of the initial assignment derived from AC-3/initial board
        if not self.consistent(initial_assignment):
             print("Inconsistency detected after AC-3 and forming initial assignment.")
             return None
        print(f"Starting backtracking with {len(initial_assignment)} pre-assigned cells.")

        # 4. Perform backtracking search starting with the pruned domains and initial assignments
        # Pass a copy of the initial assignment to backtrack if not restoring state inside it
        solution_assignment = self.backtrack(initial_assignment.copy())

        end_time = time.time() # Optional timing
        print(f"Solver finished in {end_time - start_time:.4f} seconds.")

        # 5. Format the result into a 9x9 board
        if solution_assignment:
            print("Solution found!")
            solved_board = [[0 for _ in range(9)] for _ in range(9)]
            for (r, c), value in solution_assignment.items():
                solved_board[r][c] = value
            return solved_board
        else:
            print("No solution found by backtracking.")
            return None # No solution found

# === Helper Function ===
def read_sudoku_from_file(filename):
    """Reads a Sudoku puzzle from a text file.
       Expects 9 lines, each with 9 space-separated numbers (0 for empty)."""
    try:
        with open(filename, 'r') as file:
            puzzle = []
            for r, line in enumerate(file):
                # Strip whitespace, split by space, convert to int
                row_str = line.strip().split()
                if len(row_str) != 9:
                    raise ValueError(f"Invalid number of columns ({len(row_str)}) in row {r+1}")
                row = [int(num) for num in row_str]
                puzzle.append(row)
            if len(puzzle) != 9:
                 raise ValueError(f"Invalid number of rows ({len(puzzle)}) in file")
        return puzzle
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except ValueError as e:
        print(f"Error reading file '{filename}': {e}")
        return None
    except Exception as e: # Catch other potential errors like non-integer values
        print(f"An unexpected error occurred reading file '{filename}': {e}")
        return None

# === Main Execution ===
if __name__ == "__main__":
    # --- IMPORTANT ---
    # Create a text file (e.g., "sudoku_easy.txt") in the same directory
    # as this script, or provide the full path to the file.
    # The file should contain the Sudoku puzzle like this:
    #
    # 5 3 0 0 7 0 0 0 0
    # 6 0 0 1 9 5 0 0 0
    # 0 9 8 0 0 0 0 6 0
    # 8 0 0 0 6 0 0 0 3
    # 4 0 0 8 0 3 0 0 1
    # 7 0 0 0 2 0 0 0 6
    # 0 6 0 0 0 0 2 8 0
    # 0 0 0 4 1 9 0 0 5
    # 0 0 0 0 8 0 0 7 9
    #
    # (Use 0 for empty cells)
    # --- ----------- ---

    filename = "sudoku_easy.txt" # <--- CHANGE THIS FILENAME AS NEEDED

    print(f"Attempting to read Sudoku from: {filename}")
    puzzle = read_sudoku_from_file(filename)

    if puzzle:
        print("\nInitial Puzzle:")
        for row in puzzle:
            print(" ".join(map(str, row))) # Print formatted board

        print("\nInitializing Solver...")
        try:
            solver = Sudoku_AI_solver(puzzle)

            print("\nSolving...")
            solved_board = solver.solve()

            print("\nResult:")
            if solved_board:
                print("Solved Sudoku Board:")
                for row in solved_board:
                    # Print neatly formatted solved board
                    print(" ".join(map(str, row)))
            else:
                # Solver already printed the reason (AC3 failure or no backtrack solution)
                print("Could not solve the Sudoku puzzle.")

        except ValueError as e:
             print(f"\nError initializing solver: {e}")
        except Exception as e:
             print(f"\nAn unexpected error occurred during solving: {e}")

    else:
        # read_sudoku_from_file already printed the error
        print("\nCould not read puzzle from file. Exiting.")

