import numpy as np
from dh import Joint, ForwardKinematicsDH
from inversekinematics import InverseKinematics

def main():
    Base = Joint(0.5, 0.2, 0, -90)
    Shoulder = Joint(0, 0.5, 45, 0)
    Elbow = Joint(0, 0.3, 0, -90)
    Wrist1 = Joint(0, 0.4, 90, 90)
    Wrist2 = Joint(0, 0, 0, -90)

    fk = ForwardKinematicsDH(Base, Shoulder, Elbow, Wrist1, Wrist2)
    
    ik = InverseKinematics(fk)
    desired_position = np.array([0.3, 0.4, 0.5])
    joint_angles = ik.compute_ik(desired_position)
    
    print("Ângulos das juntas para alcançar a posição desejada:")
    print(np.degrees(joint_angles))

if __name__ == "__main__":
    main()
