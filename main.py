import tkinter as tk
from tkinter import messagebox
from tkinter import *
import numpy as np
import math
from fractions import Fraction
from sympy import sympify


# This class contains the GUI implementations.
class GUI:
    def __init__(self, root, matrix):
        self.root = root
        self.matrix = matrix
        self.root.title("Elementary Matrix Calculator")
        self.root.geometry("500x500")

        # label of the title
        self.title_label = tk.Label(root, text="Elementary Matrix Calculator", font=("Arial", 16))
        self.title_label.pack(pady=10)

        # get matrix input size from user
        self.size_note = tk.Label(root, text="To create a new matrix, click 'Create Matrix' again after entering a "
                                             "new n value.")
        self.size_note.pack()
        self.size_label = tk.Label(root, text="Enter n, in which n is the matrix size in n x n matrix: ")
        self.size_label.pack()
        self.size_entry = tk.Entry(root, width=10)
        self.size_entry.pack(pady=5)

        # create a matrix button
        self.populate_button = tk.Button(root, text="Create Matrix", command=self.create_matrix)
        self.populate_button.pack(pady=10)
        self.matrix_entries = []
        self.matrix_frame = None

        # determinant button
        self.determinant_button = tk.Button(root, text="Calculate Determinant", command=self.calculate_determinant)
        self.determinant_button.pack(pady=10)

        # RREF button
        self.rref_button = tk.Button(root, text="Calculate RREF", command=self.calculate_rref)
        self.rref_button.pack(pady=10)

        # elementary matrices button
        self.elementary_button = tk.Button(root, text="Calculate Elementary Matrices", command=self.elementary_matrices)
        self.elementary_button.pack(pady=10)

        # results button
        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)

    # method allows for unique math characters such as pi, euler's number, fractions, and radicals
    def parse_input(self, value):
        safe_scope = {
            "pi": math.pi,
            "e": math.e,
            "sqrt": math.sqrt,
            "Fraction": Fraction
        }
        try:
            return sympify(value, locals=safe_scope)
        except:
            raise ValueError(f"Invalid input: {value}")

    # implementing a separate matrix creation function for the GUI
    def create_matrix(self):
        try:
            size = int(self.size_entry.get())  # gets the matrix size from user input
            if size <= 0:
                raise ValueError("Matrix size must be a positive integer.")

            self.matrix.rows = size
            self.matrix.cols = size

            # clears previous matrix frame if there was one
            if self.matrix_frame:
                self.matrix_frame.destroy()

            self.matrix_frame = tk.Frame(self.root)
            self.matrix_frame.pack(pady=10)

            # creates entry widgets for the matrix based on the user's input
            self.matrix_entries = [[tk.Entry(self.matrix_frame, width=5) for _ in range(size)] for _ in range(size)]
            for i, row in enumerate(self.matrix_entries):
                for j, entry in enumerate(row):
                    entry.grid(row=i, column=j, padx=2, pady=2)

        # error for invalid matrix size inputs
        except ValueError as e:
            messagebox.showerror("Error", f"Please enter a valid matrix size: {e}")

    def create_matrix_window(self, matrix):
        # create a new Toplevel window for the elementary matrices display
        matrix_window = Toplevel(self.root)
        matrix_window.title("Elementary Matrices Display")

        # create the canvas
        canvas = Canvas(matrix_window)
        canvas.pack(side="top", fill="both", expand=True)

        # create the horizontal scrollbar
        h_scrollbar = Scrollbar(matrix_window, orient="horizontal", command=canvas.xview)
        h_scrollbar.pack(side="bottom", fill="x")
        canvas.configure(xscrollcommand=h_scrollbar.set)

        # create a frame that will hold the matrices inside the canvas
        matrix_frame = Frame(canvas)
        canvas.create_window((0, 0), window=matrix_frame, anchor="nw")

        # displays the original matrix as a whole
        original_matrix_label = tk.Label(matrix_frame, text="Original Matrix")
        original_matrix_label.grid(row=0, column=0, padx=5, pady=5)
        original_matrix_str = self.format_matrix(self.matrix.og)
        original_matrix_text = tk.Label(matrix_frame, text=original_matrix_str, justify='left', font=("Arial", 12))
        original_matrix_text.grid(row=1, column=0, padx=5, pady=5)

        # display each elementary matrix next to the original matrix
        num_elementary_matrices = len(self.matrix.elementary_matrices)
        for idx, elem_matrix in enumerate(self.matrix.elementary_matrices):
            elementary_matrix_label = tk.Label(matrix_frame, text=f"e{idx + 1}")
            elementary_matrix_label.grid(row=0, column=idx + 1, padx=5, pady=5)
            elementary_matrix_str = self.format_matrix(elem_matrix)
            elementary_matrix_text = tk.Label(matrix_frame, text=elementary_matrix_str, justify='left',
                                              font=("Arial", 12))
            elementary_matrix_text.grid(row=1, column=idx + 1, padx=5, pady=5)

        # update so you can scroll to see everything
        matrix_frame.update_idletasks()  # makes sure all widgets are rendered
        canvas.config(scrollregion=canvas.bbox("all"))  # area of the scrollable region

    # this function reads the matrix values from the GUI
    def read_matrix(self):
        try:
            return [[self.parse_input(entry.get()) for entry in row] for row in self.matrix_entries]
        except ValueError as e:
            raise ValueError(f"Invalid input in matrix: {e}")

    def calculate_determinant(self):
        try:
            # reads matrix values from the GUI and updates the matrix
            matrix = self.read_matrix()
            self.matrix.entries = matrix  # updates here

            # calculates the determinant using previous implementation
            det = self.matrix.determinant()
            # if the determinant is the same as the integer,
            # then just return the integer version instead of the float version
            if det == int(det):
                det = int(det)

            # displays the result in the label
            self.result_label.config(text=f"Determinant: {round(det, 8)}")
        except Exception as e:
            messagebox.showerror("Error",
                                 f"Invalid matrix input. Please make sure all fields are filled with values.\n{e}")

    def calculate_rref(self):
        try:
            # same as determinant function, reads value from GUI
            matrix = self.read_matrix()
            self.matrix.og = self.read_matrix()  # updates to this as well to keep original matrix
            self.matrix.copy_matrix = self.read_matrix()
            self.matrix.entries = matrix

            # applies RREF using the previous implementation
            self.matrix.rref()

            # display the result
            formatted_matrix = self.format_matrix(self.matrix.entries)
            messagebox.showinfo("RREF Matrix", f"RREF: \n{formatted_matrix}")

        except Exception as e:
            messagebox.showerror("Error",
                                 f"Invalid matrix input. Please ensure all fields are filled with numbers.\n{e}")

    # modify the elementary_matrices method in the GUI class
    def elementary_matrices(self):
        def remove_identity_matrices(elementary_matrices):
            filtered_matrices = []
            for matrix in elementary_matrices:
                # Check if the matrix is a pure identity matrix
                is_identity = np.allclose(matrix, np.eye(matrix.shape[0]))

                # Only add if it's not a pure identity matrix
                if not is_identity:
                    filtered_matrices.append(matrix)

            return filtered_matrices

        try:
            # same as before as well
            matrix = self.read_matrix()
            self.matrix.og = self.read_matrix()  # updates to this as well to keep original matrix
            self.matrix.copy_matrix = self.read_matrix()
            self.matrix.entries = matrix

            # have to account for non-factorable matrix
            det = self.matrix.determinant()
            if det == 0:
                messagebox.showinfo("Elementary Matrices", "Matrix is singular and cannot be factored!")
            else:
                # Filter out identity matrices
                self.matrix.elementary_matrices = remove_identity_matrices(self.matrix.elementary_matrices)

                # get elementary matrices and displays them
                elementary_matrices = self.matrix.elementary_matrices
                if elementary_matrices:  # if it exists, print out, otherwise tell user
                    self.create_matrix_window(self.matrix.elementary_matrices)
                    self.matrix.elementary_matrices.clear()  # clears after printing
                else:
                    # have to call RREF here to do the process
                    self.matrix.rref()
                    # Filter out identity matrices again
                    self.matrix.elementary_matrices = remove_identity_matrices(self.matrix.elementary_matrices)
                    self.create_matrix_window(self.matrix.elementary_matrices)
                    self.matrix.elementary_matrices.clear()  # clears after printing
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while showing elementary matrices: {e}")

    def format_matrix(self, matrix):
        # helper function to format numbers
        def format_number(num):
            # check if the number is an integer
            if num % 1 == 0:
                return str(int(num))  # converts to integer if whole number
            return str(round(num, 8))

        # format matrix as a string with brackets and reduced space between values
        matrix_str = "\n"
        for row in matrix:
            row_str = "  ".join([format_number(val) for val in row])  # reduced space between elements
            matrix_str += f"  [{row_str}]\n"  # each row is enclosed in brackets

        return matrix_str


# Create a matrix class.
class Matrix:
    def __init__(self, n=2):
        self.rows = n
        self.cols = n
        # populates with 0s at the start
        self.og = [[0 for _ in range(n)] for _ in range(n)]
        self.entries = [[0 for _ in range(n)] for _ in range(n)]
        self.copy_matrix = [[0 for _ in range(n)] for _ in range(n)]
        # basically a queue of elementary matrices
        self.elementary_matrices = []

    # For the next few methods, I'm refactoring my original code so that they're more specific, specialized,
    # and general.
    def swap_rows(self, r1, r2):
        E = np.eye(self.rows)  # creates 2D array
        E[r1], E[r2] = E[r1], E[r2]  # representing the swap between any row 1 and row 2
        self.elementary_matrices.append(E)  # adds to the elementary matrices stack
        self.copy_matrix[r1], self.copy_matrix[r2] = self.copy_matrix[r2], self.copy_matrix[
            r1]  # performs the actual swap

    def scale(self, pos, factor):
        E = np.eye(self.rows)
        # pos represents the position of the element in the matrix, since factor * 1 is always factor,
        # just set equal to factor
        # in the code, we're dividing by factor, so we have to get the reciprocal to get the actual factor value
        E[pos, pos] = 1 / factor
        self.elementary_matrices.append(E)
        # list comprehension that multiplies factor to all elements in the row that pos was from
        self.copy_matrix[pos] = [x * factor for x in self.copy_matrix[pos]]

    def add_multiple(self, row1, row2, factor):
        E = np.eye(self.rows)
        # for elementary matrices, adding a scalar multiple of a row to another row is represented as
        # factor * 1 in the position of a, b, in which a is the row position and b is the column position.
        # therefore, at the element of row1 and row2, we can just set it to factor.
        E[row1, row2] = factor
        self.elementary_matrices.append(E)
        # list comprehension that adds a multiple of the second row to the first row for every element
        self.copy_matrix[row1] = [x - factor * y for x, y in zip(self.copy_matrix[row1], self.copy_matrix[row2])]

    # This method calculates the determinant.
    def determinant(self, lead=0):
        # if the matrix is 1x1
        if self.rows == 1:
            return self.entries[0]

        # if the matrix is 2x2
        if self.rows == 2:
            return self.entries[0][0] * self.entries[1][1] - self.entries[0][1] * self.entries[1][0]

        # creating a triangle to multiply the diagonal
        # after every row and column, stop the recursion (nxn matrix so only need to use one)
        # multiply the diagonal in the process
        if lead >= self.rows:
            det = 1
            for i in range(self.rows):
                det *= self.entries[i][i]
            return round(det, 2)  # rounds to two decimal places

        # finding the pivot/first nonzero entry
        i = lead
        while self.entries[i][lead] == 0:
            i += 1
            # if there is no pivot in this column, end current recursion and move to the next
            if i == self.rows:
                return self.determinant(lead + 1)

        # swap to move the pivot row to the current row if necessary
        if i != lead:
            self.entries[i], self.entries[lead] = self.entries[lead], self.entries[i]
            det = -1
        else:
            det = 1

        # eliminating the entries in other rows by using the factor of the pivot entry
        for i in range(self.rows):
            if i != lead:
                factor = self.entries[i][lead] / self.entries[lead][lead]
                self.entries[i] = [x - factor * y for x, y in zip(self.entries[i], self.entries[lead])]

        # recursive statement for the sub-matrix (adds one to lead because that controls which column it's operating on)
        return det * self.determinant(lead + 1)

    # This method applies RREF to the matrix.
    def rref(self, lead=0):
        # Base case: stop if we've processed all columns or rows
        if lead >= self.rows or lead >= self.cols:
            return True

        # find the pivot row for the current column
        pivot_row = None
        for i in range(lead, self.rows):
            if self.copy_matrix[i][lead] != 0:
                pivot_row = i
                break

        # if no pivot found in this column, move to next column
        if pivot_row is None:
            return self.rref(lead + 1)

        # swap rows to bring pivot to the current lead row if necessary
        if pivot_row != lead:
            self.swap_rows(pivot_row, lead)

        # scale the pivot row to make the pivot 1
        pivot = self.copy_matrix[lead][lead]
        if pivot != 1:
            self.scale(lead, 1 / pivot)

        # eliminate entries in other rows
        for i in range(self.rows):
            if i != lead:
                factor = self.copy_matrix[i][lead]
                if factor != 0:
                    self.add_multiple(i, lead, factor)

        # assigns at the end
        self.entries = self.copy_matrix

        # recursively process the next column
        return self.rref(lead + 1)


# driver code
if __name__ == "__main__":
    matrix = Matrix()
    root = tk.Tk()
    app = GUI(root, matrix)
    root.mainloop()
