import numpy as np
from dh import Joint, ForwardKinematicsDH
from inversekinematics import InverseKinematics
import time 

def main():
    # Definição das juntas do manipulador, com os parâmetros de Denavit-Hartenberg
    # d (metros), a (metros), theta (graus), alpha (graus)
    Base = Joint(3, 0, 0, -90)
    Shoulder = Joint(0, 2, -60, 0)
    Elbow = Joint(0, 2, 90, 0)
    Wrist1 = Joint(0, 0, -30, 90)
    Wrist2 = Joint(1.5, 0, 0, 0)

    # Cálculo da cinemática direta
    fk = ForwardKinematicsDH(Base, Shoulder, Elbow, Wrist1, Wrist2)
    
    print("End effector = ", fk.compute_end_effector())

    # Cálculo da cinemática inversa
    ik = InverseKinematics(fk)
    desired_position = np.array([1, 0.5, 0.5])
    joint_angles = ik.compute_ik(desired_position)
    
    print("Ângulos das juntas para alcançar a posição desejada:")
    print(np.degrees(joint_angles))
    ik.error_Counter()

if __name__ == "__main__":
    inicio = time.time()
    main()
    fim = time.time()
    print("Tempo de execução: ", fim - inicio)
