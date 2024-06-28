import numpy as np

#Cinemática direta com parâmetros de Denavit-Hartenberg

class Joint:
    def __init__(self, d, a, theta, alpha):
        self.d = d
        self.a = a
        self.theta = np.radians(theta)
        self.alpha = np.radians(alpha)
    
    def dh_matrix(self):
        """
        Calcula a matriz de transformação DH para os parâmetros da junta.
        """
        return np.array([
            [np.cos(self.theta), -np.sin(self.theta) * np.cos(self.alpha), np.sin(self.theta) * np.sin(self.alpha), self.a * np.cos(self.theta)],
            [np.sin(self.theta), np.cos(self.theta) * np.cos(self.alpha), -np.cos(self.theta) * np.sin(self.alpha), self.a * np.sin(self.theta)],
            [0, np.sin(self.alpha), np.cos(self.alpha), self.d],
            [0, 0, 0, 1]
        ])

class ForwardKinematicsDH:
    def __init__(self, *joints):
        self.joints = list(joints)
    
    def compute_end_effector(self):
        end_effector_matrix = np.identity(4)
        for joint in self.joints:
            transformation = joint.dh_matrix()
            end_effector_matrix = np.dot(end_effector_matrix, transformation)
        return end_effector_matrix

    def compute_jacobian(self):
        n = len(self.joints)
        J = np.zeros((6, n))
        
        T = np.identity(4)
        positions = [np.array([0, 0, 0])]
        z_axes = [np.array([0, 0, 1])]
        
        for joint in self.joints:
            T = np.dot(T, joint.dh_matrix())
            positions.append(T[:3, 3])
            z_axes.append(T[:3, 2])
        
        end_effector_position = positions[-1]
        
        for i in range(n):
            J[:3, i] = np.cross(z_axes[i], (end_effector_position - positions[i]))
            J[3:, i] = z_axes[i]
        
        return J

