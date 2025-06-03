import numpy as np
from numpy.linalg import det, inv, norm
import sympy as sp  #biblioteka umozliwiajaca operacje symboliczne

"""### Losowe generowanie macierzy"""

np.set_printoptions(precision=5, suppress=False, floatmode='fixed')
tolerance=1e-20
n=5

A=np.random.randint(low=0, high=10, size=(n, n)) #macierz, na której będziemy wykonywać wszelkie operacje
B=np.random.randint(low=0, high=10, size=(n, n)) #macierz do testu mnożenia

print("Macierz A: ")
print(A)
print("Macierz B: ")
print(B)

Z=A
k=0

"""### Postać schodkowa"""

def schodkowa(A):
    k=0
    A = A.copy().astype(float) #tworzy kopię i operacje wykonujemy na niej
    n = A.shape[0]
    if np.all(A == 0): #jeśli macierz jest wypełniona samymi zerami, to jest w postaci schodkowej
        return A
    current_row = 0
    for col in range(n): #przechodzimy przez kolejne kolumny
        pivot_row = None
        for row in range(current_row, n): #szukamy pierwszego niezerowego elementu w kolumnie)
            if abs(A[row, col]) > tolerance:  #używamy tolerancji dla błędów numerycznych
                pivot_row = row
                break
        if pivot_row is None: #jeśli cała kolumna jest zerowa, przechodzimy do następnej
            continue
        if pivot_row != current_row: #aby mieć niezerowy element w pierwszym wierszu, zamieniamy wiersze
            A[[current_row, pivot_row]] = A[[pivot_row, current_row]]
            k+=1
        for row in range(current_row + 1, n): #stosujemy eliminację w kolumnie
            if abs(A[row, col]) > tolerance:
                multiplier = A[row, col] / A[current_row, col]
                A[row, col:] = A[row, col:] - multiplier * A[current_row, col:]
        current_row += 1
        if current_row >= n:
            break

    return A,k

aret,kret=schodkowa(A)
print("Postac schodkowa macierzy: ")
print(aret)

"""### Postać zredukowana"""

def zredukowana(A):
    B=aret.transpose()
    bret,kret1=schodkowa(B)
    return bret

bret = zredukowana(aret)
print("Postac zredukowana macierzy: ")
print(bret)

"""### Wyznacznik macierzy"""

def wyznacznik(A):
    n=A.shape[0]
    m=1
    m=float(m)
    for i in range(n):
        m*=float(A[i,i])
    m*=(-1)**(kret%2)
    return m

print("Wyznacznik macierzy: ")
print(wyznacznik(aret))

"""### Ślad macierzy"""

def slad(A):
    q=0
    n = A.shape[0]
    for i in range(n):
        q+=A[i,i]
    return q

print("Slad macierzy: ")
print(slad(A))

"""### Rząd macierzy"""

def rzad(A):
    n=A.shape[0]
    if np.all(A == 0):
        return 0
    a=0
    i=0
    for i in range(n):
        if abs(A[i,i])>tolerance:
            a+=1
    return a

print("Rzad macierzy: ")
print(rzad(bret))

"""### Macierz transponowana"""

def transpozycja(A):
  A = A.copy().astype(float)
  m=A.shape[0]  #liczba wierszy macierzy
  n=A.shape[1]  #liczba kolumn macierzy
  X = np.zeros((n,m), dtype=float) #pusta macierz o wymiarach n x m

  for i in range (m):
    for j in range (n):
      X[j,i]=A[i,j] #operacja transponowania

  return X

print("Macierz transponowana: ")
print(transpozycja(A))

"""### Mnożenie macierzy"""

def mnozenie(A,B):
  mA=A.shape[0]
  nA=A.shape[1]
  mB=B.shape[0]
  nB=B.shape[1]
  if nA!=mB: #warunek konieczny mnozenia macierzy
    print ("Tych macierzy nie da sie pomnozyc")
    return ""
  X=np.zeros((mA,nB), dtype=float) #pusta macierz o wymiarach wiersze A x kolumny B
  for i in range (mA):
    for j in range (nB):
      for k in range (nA):
        X[i,j]+=A[i,k]*B[k,j] #mnozenie macierzy
  return X

print ("Wynik mnozenia macierzy: ")
print (mnozenie(A,B))

"""### Macierz odwrotna"""

def minor(A,i,j):  #funkcja pomocnicza, wyznaczajaca minory
  A = A.copy().astype(float)
  M = np.delete(A, i, axis=0)  #usuniecie i-tego wiersza
  M = np.delete(M, j, axis=1)  #usuniecie j-tej kolumny
  return M

def odwrotna(A):
  A = A.copy().astype(float)
  m=A.shape[0]
  n=A.shape[1]  #wymiary macierzy A

  if m!=n:  #sprawdzanie zgodnosci wymiarow
    print ("Macierz odwrotna nie istnieje")
    return ""

  if abs(wyznacznik(A))<tolerance:
    print ("Macierz odwrotna nie istnieje") #warunek na odwracalnosc macierzy
    return ""

  X = np.zeros((m,n), dtype=float) #pusta macierz, wynik odwrotnosci
  for i in range (n):
    for j in range (m):
      X[i,j]=((-1)**(i+j))*wyznacznik(minor(A,i,j))  # macierz stowarzyszona

  X=transpozycja(X)
  X=X/wyznacznik(A)
  return X

print ("Macierz odwrotna: ")
print (odwrotna(A))

"""### Wielomian charakterystyczny"""

def wielomian_charakterystyczny(A):
    A = A.copy().astype(float)
    m= A.shape[0]
    n=A.shape[1]  #wymiary macierzy A
    if m != n:
        print("Macierz nie jest kwadratowa – nie można wyznaczyć wielomianu.")
        return ""

    λ = sp.symbols('λ')  #symbol lambda, uzywany do zapisu wielomianu
    X = sp.Matrix(A)  #konwersja na macierz w sympy
    I = sp.zeros(n,n)
    for i in range (n):
      I[i,i]= 1  #generowanie macierzy jednostkowej

    X = X - λ * I  #w_A(\lambda)=\det(A-\lambda I),
    wyzn= sp.simplify(X.det())  #wyznacznik macierzy X, uproszczenie
    wielomian = sp.Poly(wyzn, λ)  #utworzenie wielomianu ze zmienna \lambda
    return wielomian

print("Wielomian charakterystyczny:")
print(wielomian_charakterystyczny(A).as_expr())

"""### Sprawdzanie przynależności do grup liniowych"""

def is_GL(A):
    """
    Sprawdzenie czy macierz należy do GL(n)
    Macierz należy do GL(n) jeśli jest odwracalna
    """
    return abs(wyznacznik(A)) > tolerance

def is_SL(A):
    """
    Sprawdzenie czy macierz należy do SL(n)
    Macierz należy do SL(n) jeśli należy do GL(n) i det = 1
    """
    return abs(wyznacznik(A) - 1) < tolerance

def is_orthogonal(A):
    """
    Sprawdzenie czy macierz należy do O(n)
    Macierz należy do O(n) jeśli A^{-1} = A^T
    """
    return np.allclose(A @ A.T, np.eye(A.shape[0]), atol=tolerance)

def is_SO(A):
    """
    Sprawdzenie czy macierz należy do SO(n)
    j.w., należy do O(n) oraz det =1
    """
    return is_orthogonal(A) and abs(wyznacznik(A) - 1) < tolerance

def is_unitary(A):
    """
    Sprawdzenie czy macierz należy do U(n)
    Macierz należy do U(n) jeśli A^{-1} = A^*
    """
    return np.allclose(A @ A.conj().T, np.eye(A.shape[0]), atol=tolerance)

def is_SU(A):
    """
    Sprawdzenie czy macierz należy do SU(n)
    Należy do SU(n), jeśli należy do U(n) i det = 1
    """
    return is_unitary(A) and abs(wyznacznik(A) - 1) < tolerance

def is_symplectic(A):
    """
    Sprawdzenie czy macierz należy do grupy symplektycznej Sp(2n)
    Macierz M jest symplektyczna jeśli jest macierzą niezdeg. antysym. formy dwuliniowej, czyli M^T J M = J gdzie J = [0 I; -I 0]
    """
    n = A.shape[0]
    if n % 2 != 0:
        return False  # Wymiar musi być parzysty

    # Tworzymy macierz J
    m = n // 2
    J = np.block([[np.zeros((m, m)), np.eye(m)],
                  [-np.eye(m), np.zeros((m, m))]])

    return np.allclose(A.T @ J @ A, J, atol=tolerance)

def check_matrix_groups(A):
    """
    Polecenie zwraca booleanowy słownik, gdzie kluczami są grupy liniowe
    """
    results = {
        'GL': is_GL(A),
        'SL': is_SL(A),
        'O': is_orthogonal(A),
        'SO': is_SO(A),
        'U': is_unitary(A),
        'SU': is_SU(A),
        'Sp': is_symplectic(A)
    }
    return results

"""### Przykłady"""

identity = np.eye(2)
rotation = np.array([[0.6, -0.8], [0.8, 0.6]])  # SO(2)
unitary = np.array([[1j,1],[1,1j]]) / np.sqrt(2) # SU(2)
symplectic = np.array([[1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])  # niesymplekt.

print("id:")
print(check_matrix_groups(identity))

print("SO:")
print(check_matrix_groups(rotation))

print("U:")
print(check_matrix_groups(unitary))

print("Sp:")
print(check_matrix_groups(symplectic))

print("A:")
print(check_matrix_groups(A))

