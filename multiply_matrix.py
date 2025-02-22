import sympy as sp

''''
Este script exemplifica a multiplicação de matrizes de transformação homogênea

O intuito foi apenas para ver o resultado da matriz homogênea de um robô com 5 juntas
'''

# Definindo as variáveis simbólicas
theta1, theta2, theta3, theta4, theta5 = sp.symbols('theta1 theta2 theta3 theta4 theta5')
alpha1, alpha2, alpha3, alpha4, alpha5 = sp.symbols('alpha1 alpha2 alpha3 alpha4 alpha5')
a1, a2, a3, a4, a5 = sp.symbols('a1 a2 a3 a4 a5')
d1, d2, d3, d4, d5 = sp.symbols('d1 d2 d3 d4 d5')

# Definindo a matriz de transformação homogênea
def dh_transform(theta, alpha, a, d):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta), sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0, sp.sin(alpha), sp.cos(alpha), d],
        [0, 0, 0, 1]
    ])

# Definindo as matrizes de transformação homogênea para cada junta
T01 = dh_transform(theta1, alpha1, a1, d1)
T12 = dh_transform(theta2, alpha2, a2, d2)
T23 = dh_transform(theta3, alpha3, a3, d3)
T34 = dh_transform(theta4, alpha4, a4, d4)
T45 = dh_transform(theta5, alpha5, a5, d5)

# Multiplicando as matrizes para obter T05 sem simplificação
T05 = T01 * T12 * T23 * T34 * T45

sp.pprint(T05[:3][-1])
