import math
from pathlib import Path

A = []
B = []

def extract_coefficient(equation, start, end):
    coef = equation[start:end].strip()

    if coef == '' or coef == '+':
        return 1.0
    elif coef == '-':
        return -1.0
    else:
        try:
            return float(coef)
        except ValueError:
            return 1.0

def load_system(path: Path) -> tuple[list[list[float]], list[float]]:
    with open(path, 'r') as file:
        for line in file:
            equation, result = line.split('=')
            equation = equation.replace(' ', '')
            result = float(result.strip())

            x = equation.find('x')
            y = equation.find('y')
            z = equation.find('z')

            coef_x = extract_coefficient(equation, 0, x)
            coef_y = extract_coefficient(equation, x + 1, y)
            coef_z = extract_coefficient(equation, y + 1, z)

            A.append([coef_x, coef_y, coef_z])
            B.append(result)

    return A, B

def determinant(matrix: list[list[float]]) -> float:
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )

def norm(vector: list[float]) -> float:
    return math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

def trace(matrix: list[list[float]]) -> float:
    return matrix[0][0] + matrix[1][1] + matrix[2][2]

def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    result = [0] * len(matrix)

    for i in range(len(matrix)):
        result[i] = sum(matrix[i][j] * vector[j] for j in range(len(vector)))

    return result

def replace_column(matrix: list[list[float]], column: int, vector: list[float]) -> list[list[float]]:
    new_matrix = [row[:] for row in matrix]
    for i in range(len(matrix)):
        new_matrix[i][column] = vector[i]
    return new_matrix

def cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det_A = determinant(A)

    if det_A == 0:
        raise ValueError("The linear system of equations can not be solved with Cramer because the determinant is 0.")

    det_Ax = determinant(replace_column(matrix, 0, vector))
    det_Ay = determinant(replace_column(matrix, 1, vector))
    det_Az = determinant(replace_column(matrix, 2, vector))

    x = det_Ax / det_A
    y = det_Ay / det_A
    z = det_Az / det_A

    return [x, y, z]

def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    return [[matrix[r][c] for c in range(len(matrix)) if c != j] for r in range(len(matrix)) if r != i]

def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    cofactors = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix)):
            minor_matrix = minor(matrix, i, j)
            cofactor_value = ((-1) ** (i + j)) * determinant(minor_matrix)
            row.append(cofactor_value)
        cofactors.append(row)
    return cofactors

def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    cofactors = cofactor(matrix)
    return [[cofactors[j][i] for j in range(len(matrix))] for i in range(len(matrix))]


def inverse(matrix: list[list[float]]) -> list[list[float]]:
    det_A = determinant(matrix)
    if det_A == 0:
        raise ValueError("The matrix is not invertible because the determinant is 0.")

    adjugate_matrix = adjoint(matrix)
    return [[adjugate_matrix[i][j] / det_A for j in range(len(matrix))] for i in range(len(matrix))]

def inversion_solution(matrix: list[list[float]], vector: list[float]) -> list[float]:
    inverse_A = inverse(matrix)
    return multiply(inverse_A, vector)

def print_matrix(Title, M):
    print(Title)
    for row in M:
        print([x for x in row])


# Main-ul:
A, B = load_system(Path("system.txt"))
B_matrix = [[b] for b in B]
print_matrix("The matrix A:", A)
print_matrix("The result's matrix B:", B_matrix)

det_A = determinant(A)
print(f"A's determinant: {det_A}")

trace_A = trace(A)
print(f"A's trace: {trace_A}")

norm_B = norm(B)
print(f"B's norm: {norm_B}")

transpose_A = transpose(A)
print_matrix("The transpose matrix of A: ", transpose_A)

multiply_result = multiply(A, B)
multiply_result_matrix =  [[linie] for linie in multiply_result]
print_matrix("The multiplication of A and B:", multiply_result_matrix)

solution = cramer(A, B)
print(f"The Cramer solution for the system: {solution}")

solution = inversion_solution(A, B)
print(f"The Inversion solution for the system: {solution}")