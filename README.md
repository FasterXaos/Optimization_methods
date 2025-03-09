# Function Minimization Application

This is a Python application built using **PyQt5** for the graphical user interface (GUI) and **matplotlib** for visualizing the function minimization process. The application allows users to minimize a selected mathematical function using different optimization methods, such as:

- Nelder-Mead Method
- Gradient Descent Method
- Conjugate Gradient Method
- Newton's Method

## Features

- Choose between two predefined functions to minimize:
  - Rosenbrock's function: `f1(x) = 100(x₂ - x₁²)² + 5(1 - x₁)²`
  - The Beale function: `f2(x) = (x₁² + x₂ - 11)² + (x₁ + x₂² - 7)²`

- Set initial values for `x₁` and `x₂` and perform minimization using one of the four optimization methods.

- The optimization methods:
  - **Nelder-Mead**: A simplex-based method for optimization.
  - **Gradient Descent**: Iterative optimization based on the gradient of the function.
  - **Conjugate Gradient**: Uses the conjugate gradient method to find the minimum.
  - **Newton's Method**: A second-order method using both gradient and Hessian matrix.

- Visualization: A contour plot of the function is displayed, and the minimum point is marked on the plot.

## Installation Requirements

To run this application, you'll need to install the following Python packages:
- `PyQt5`
- `numpy`
- `matplotlib`
- `sympy`

You can install them using pip:
```bash
pip install PyQt5 numpy matplotlib sympy
```
## How to Run

Execute the script to launch the application:
```bash
Minimization_app.pyw
```
## GUI Elements

- **Function Selector**: Choose the function to minimize.
- **Initial Value Inputs**: Set the initial values for `x₁` and `x₂`.
- **Method Selector**: Select the optimization method (Nelder-Mead, Gradient Descent, Conjugate Gradient, or Newton's Method).
- **Minimize Button**: Click to start the minimization process.
- **Visualization Panel**: Displays a contour plot of the selected function with the computed minimum point highlighted.

## Example Usage

1. Select a function (Rosenbrock's or Beale's).
2. Input initial values for `x₁` and `x₂`.
3. Choose an optimization method.
4. Click the **"Minimize"** button.
5. The application will perform the minimization and display the result on the contour plot.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
