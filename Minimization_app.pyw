import sys
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QComboBox, QLineEdit, QPushButton, QHBoxLayout, QFrame)
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MinimizationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        mainLayout = QHBoxLayout()
        mainLayout.setSpacing(0)

        leftPanelLayout = QVBoxLayout()
        leftPanelContainer = QFrame()
        leftPanelContainer.setLayout(leftPanelLayout)
        leftPanelContainer.setFixedWidth(250)
        leftPanelLayout.setSpacing(10)
        
        vectorInputLayout = QHBoxLayout()
        vectorInputLayout.setSpacing(5)
        
        self.functionLabel = QLabel("Выберите функцию:")
        self.functionSelector = QComboBox()
        self.functionSelector.addItems(["f1(x) = 100(x₂ - x₁²)² + 5(1 - x₁)²", 
                                     "f2(x) = (x₁² + x₂ - 11)² + (x₁ + x₂² - 7)²"])
        
        self.x1Label = QLabel("x1=")
        self.x1Input = QLineEdit("0.0")
        self.x2Label = QLabel("x2=")
        self.x2Input = QLineEdit("0.0")

        vectorInputLayout.addWidget(self.x1Label)
        vectorInputLayout.addWidget(self.x1Input)
        vectorInputLayout.addSpacing(10)
        vectorInputLayout.addWidget(self.x2Label)
        vectorInputLayout.addWidget(self.x2Input)
        
        self.methodLabel = QLabel("Выберите метод:")
        self.methodSelector = QComboBox()
        self.methodSelector.addItems(["Метод деформируемого многогранника",
                                    "Метод градиентного спуска",
                                    "Метод сопряженных градиентов",
                                    "Метод Ньютона"])
        
        self.minimizeButton = QPushButton("Минимизировать")
        self.minimizeButton.clicked.connect(self.performMinimization)
        
        self.separatorLine1 = QFrame()
        self.separatorLine1.setFrameShape(QFrame.HLine)
        self.separatorLine1.setFrameShadow(QFrame.Sunken)

        self.separatorLine2 = QFrame()
        self.separatorLine2.setFrameShape(QFrame.HLine)
        self.separatorLine2.setFrameShadow(QFrame.Sunken)
        
        leftPanelLayout.addWidget(self.functionLabel)
        leftPanelLayout.addWidget(self.functionSelector)
        leftPanelLayout.addLayout(vectorInputLayout)
        leftPanelLayout.addWidget(self.separatorLine1)
        leftPanelLayout.addWidget(self.methodLabel)
        leftPanelLayout.addWidget(self.methodSelector)
        leftPanelLayout.addWidget(self.minimizeButton)
        leftPanelLayout.addWidget(self.separatorLine2)
        
        self.ImageLabelFNAF = QLabel()
        PixmapFNAF  = QPixmap("fnaf.jpg")
        self.ImageLabelFNAF.setPixmap(PixmapFNAF )
        self.ImageLabelFNAF.setScaledContents(True)
        leftPanelLayout.addWidget(self.ImageLabelFNAF)

        self.ImageLabelNEKOPARA = QLabel()
        PixmapNEKOPARA  = QPixmap("NEKOPARA.jpg")
        self.ImageLabelNEKOPARA.setPixmap(PixmapNEKOPARA )
        self.ImageLabelNEKOPARA.setScaledContents(True)
        self.ImageLabelNEKOPARA.setFixedHeight(100)
        leftPanelLayout.addWidget(self.ImageLabelNEKOPARA)
        
        leftPanelLayout.addStretch()
        
        self.figure, self.axis = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        
        mainLayout.addWidget(leftPanelContainer)
        mainLayout.addWidget(self.canvas)
        self.setLayout(mainLayout)
        self.setWindowTitle("Минимизация функций")
        self.setGeometry(100, 100, 1000, 500)

        x1, x2 = sp.symbols('x1 x2')
        self.functions = [
            (lambda x: 100 * (x[1] - x[0]**2)**2 + 5 * (1 - x[0])**2,  
             [sp.diff(100 * (x2 - x1**2)**2 + 5 * (1 - x1)**2, var) for var in (x1, x2)]),
            
            (lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2,  
             [sp.diff((x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2, var) for var in (x1, x2)])
        ]

    def computeObjectiveFunction(self, x):
        functionIndex = self.functionSelector.currentIndex()
        return self.functions[functionIndex][0](x)
    
    def computeGradient(self, x):
        functionIndex = self.functionSelector.currentIndex()
        x1, x2 = sp.symbols('x1 x2')
        gradientFunctions = self.functions[functionIndex][1]
        gradientValues = [gradientFunctions[0].subs({x1: x[0], x2: x[1]}),
                          gradientFunctions[1].subs({x1: x[0], x2: x[1]})
        ]
        return np.array(gradientValues, dtype=float)
        
    def performMinimization(self):
        x1Initial = float(self.x1Input.text())
        x2Initial = float(self.x2Input.text())
        x0 = np.array([x1Initial, x2Initial])
        
        methodIndex = self.methodSelector.currentIndex()

        if methodIndex == 0:
            result = self.nelderMeadOptimization(x0)
        elif methodIndex == 1:
            result = self.gradientDescentOptimization(x0)
        elif methodIndex == 2:
            result = self.conjugateGradient(x0)
        elif methodIndex == 3:
            result = self.newtonOptimization(x0)
        else:
            return
        
        self.plotFunction(result)

    def nelderMeadOptimization(self, x0, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, tolerance=1e-9, maxIterations=10000):
        objectiveFunction = self.computeObjectiveFunction
        n = len(x0)
        simplex = np.vstack([x0] + [x0 + 0.05 * np.eye(1, n, i)[0] for i in range(n)])
        
        for iteration in range(maxIterations):
            simplex = np.array(sorted(simplex, key=objectiveFunction))
            centroid = np.mean(simplex[:-1], axis=0)
            reflectionPoint = centroid + alpha * (centroid - simplex[-1])
            
            if objectiveFunction(reflectionPoint) < objectiveFunction(simplex[0]):
                expansionPoint = centroid + gamma * (reflectionPoint - centroid)
                simplex[-1] = expansionPoint if objectiveFunction(expansionPoint) < objectiveFunction(reflectionPoint) else reflectionPoint
            elif objectiveFunction(reflectionPoint) < objectiveFunction(simplex[-2]):
                simplex[-1] = reflectionPoint
            else:
                contractionPoint = centroid + rho * (simplex[-1] - centroid)
                if objectiveFunction(contractionPoint) < objectiveFunction(simplex[-1]):
                    simplex[-1] = contractionPoint
                else:
                    simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
            
            print(f"Iteration {iteration}: x = {simplex[0]}")
            
            if np.max(np.abs(simplex - simplex[0])) < tolerance and np.std([objectiveFunction(x) for x in simplex]) < tolerance:
                break
        
        return simplex[0]

    def gradientDescentOptimization(self, x0, momentumCoefficient = 0.9, tolerance=1e-8, maxIterations=10000):
        currentPoint = np.array(x0, dtype=float)
        velocity = np.zeros_like(currentPoint)

        for iteration in range(maxIterations):
            gradientVector = np.array(self.computeGradient(currentPoint))
            gradientVector = np.clip(gradientVector, -10, 10)
            alpha = 0.01 / (1 + iteration / 100)

            velocity = momentumCoefficient * velocity + alpha * gradientVector
            currentPoint = currentPoint - velocity

            print(f"Iteration {iteration}: x = {currentPoint}")

            if np.linalg.norm(gradientVector) < tolerance:
                break

        return currentPoint


    def conjugateGradient(self, x0, tolerance=1e-8, maxIterations=10000):
        currentPoint = np.array(x0, dtype=float)
        gradientVector = self.computeGradient(currentPoint)
        searchDirection = -gradientVector

        for iteration  in range(maxIterations):
            if np.linalg.norm(gradientVector) < tolerance:
                break

            stepSize = self.lineSearch(currentPoint, searchDirection)
            newPoint = currentPoint + stepSize * searchDirection
            newGradientVector = self.computeGradient(newPoint)

            # β (Fletcher-Reeves)
            beta = np.dot(newGradientVector, newGradientVector) / np.dot(gradientVector, gradientVector)
            searchDirection = -newGradientVector + beta * searchDirection

            currentPoint, gradientVector = newPoint, newGradientVector
            print(f"Iteration {iteration }: x = {currentPoint}")

        return currentPoint

    def lineSearch(self, currentPoint, searchDirection, tolerance=1e-5, maxIterations=10000):
        goldenRatio = (1 + np.sqrt(5)) / 2
        lowerBound, upperBound = -10, 10
        leftPoint = upperBound - (upperBound - lowerBound) / goldenRatio
        rightPoint = lowerBound + (upperBound - lowerBound) / goldenRatio
        
        functionLeft = self.computeObjectiveFunction(currentPoint + leftPoint * searchDirection)
        functionRight = self.computeObjectiveFunction(currentPoint + rightPoint * searchDirection)

        for _ in range(maxIterations):
            if abs(upperBound - lowerBound) < tolerance:
                break
            
            if functionLeft < functionRight:
                upperBound, rightPoint = rightPoint, leftPoint
                leftPoint = upperBound - (upperBound - lowerBound) / goldenRatio
                functionRight, functionLeft = functionLeft, self.computeObjectiveFunction(currentPoint + leftPoint * searchDirection)
            else:
                lowerBound, leftPoint = leftPoint, rightPoint
                rightPoint = lowerBound + (upperBound - lowerBound) / goldenRatio
                functionLeft, functionRight = functionRight, self.computeObjectiveFunction(currentPoint + rightPoint * searchDirection)

        return (lowerBound + upperBound) / 2

    def newtonOptimization(self, x0, tolerance=1e-8, maxIterations=100):
        currentPoint = np.array(x0, dtype=float)

        for iteration in range(maxIterations):
            gradientVector = self.computeGradient(currentPoint)
            if np.linalg.norm(gradientVector) < tolerance:
                break

            hessianMatrix = self.computeHessian(currentPoint)
            try:
                stepDirection = np.linalg.solve(hessianMatrix, -gradientVector)  # H * delta_x = -grad
            except np.linalg.LinAlgError:
                stepDirection = -gradientVector

            currentPoint = currentPoint + stepDirection
            print(f"Iteration {iteration}: x = {currentPoint}")
        return currentPoint
    
    def computeHessian(self, x):
        selectedFunctionIndex = self.functionSelector.currentIndex()
        x1, x2 = sp.symbols('x1 x2')
        
        symbolicFunction = self.functions[selectedFunctionIndex][0]([x1, x2])
        hessianMatrix = sp.Matrix([[sp.diff(sp.diff(symbolicFunction, var1), var2) for var2 in (x1, x2)] for var1 in (x1, x2)])

        hessianValues = np.array(hessianMatrix.subs({x1: x[0], x2: x[1]}), dtype=float)
        return hessianValues
    
    def plotFunction(self, minimumPoint):
        self.axis.clear()
        functionIndex = self.functionSelector.currentIndex()
        x1Grid, x2Grid = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
        
        if functionIndex == 0:
            z = 100 * (x2Grid - x1Grid**2)**2 + 5 * (1 - x1Grid)**2
        else:
            z = (x1Grid**2 + x2Grid - 11)**2 + (x1Grid + x2Grid**2 - 7)**2
        
        self.axis.contourf(x1Grid, x2Grid, z, levels=150, cmap='jet')
        self.axis.plot(minimumPoint[0], minimumPoint[1], 'ro', markersize=8, label=str(minimumPoint))
        self.axis.legend()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MinimizationApp()
    window.show()
    sys.exit(app.exec_())
