# Solving PDEs using Physics Informed Neural Networks

# Table of Contents
1. [What is PINN ?](#introduction)
2. [First approach - Lagaris](#first_approach)
3. [Second approach](#second_approach)
4. [Requirements](#requirements)

# What is PINN ? <a name="introduction"></a>
Physics Informed Neural Network is a method for solving partial differential equations using neural networks and physical contraints. Such contraints are imposed using a loss function, which can be expressed in several ways, depending on the PINN model adopted. In this project we used 2 approachs and we consider the PDE defined as:

$Q(\psi, \nabla\psi, \Delta\psi, . . . )(x_1,x_2,...,x_n,t) = 0 $, inside $\Omega$

$R(\psi, \nabla\psi, \Delta\psi, . . . )(x_1,x_2,...,x_n,t) = f(x_1,x_2,...,x_n,t) $, in $\partial \Omega$

# First approach - Time independent <a name="first_approach"></a>
In that case the solution is constructed such that:

$\psi=F \cdot NN+A$

where NN is the neural network output, F is any function that is non-zero inside the domain and zero at the boundary and A is a function satisfying the boundary conditions. Thus, the loss function is defined using the L2 norm and the residual of the PDE:

$L = {\frac{1}{N_{samples}}\lVert Q(\psi, \nabla\psi, \Delta\psi, . . . ) \rVert}_2^2$ 

## Helmholtz equation
PINN solution             |  Squared error
:-------------------------:|:-------------------------:
<img src="./Helmholtz/Images/approximated_helmholtz.png?raw=true" width="100%">|<img src="./Helmholtz/Images/squared_error_helmholtz.png?raw=true" width="100%"> 




# Second approach - Time dependent <a name="second_approach"></a>
In that case the solution is the output of the neural network and the loss function is defined as:

$L = L_{inside}+L_{boundary}+L_{initial}$ 

Where

$L_{inside} = \frac{1}{N_{in}}{\lVert Q(\psi, \nabla\psi, \Delta\psi, . . . ) \rVert}^{2}_2$, inside $\Omega$

$L_{boundary} = \frac{1}{N_{bound}}{\lVert Q(\psi, \nabla\psi, \Delta\psi, . . . ) \rVert}^{2}_2$, in $\partial \Omega$

$L_{initial} = \frac{1}{N_{initial}}{\lVert Q(\psi, \nabla\psi, \Delta\psi, . . . ) \rVert}^{2}_2$, in $t=0$

## Taylor Green Vortex

PINN solution             |  Squared error
:-------------------------:|:-------------------------:
<img src="./Taylor_green_vortex/Images/taylor_green_vortex_p.gif?raw=true" width="100%">|<img src="./Taylor_green_vortex/Images/squared_error_p.gif?raw=true" width="100%"> 

## Mild slope equation
PINN solution 2d            |  PINN solution 3d
:-------------------------:|:-------------------------:
<img src="./Mild_slope/Images/mild_slope_animation2d.gif?raw=true" width="100%">|<img src="./Mild_slope/Images/mild_slope_animation3d.gif?raw=true" width="85%"> 

# Requirements  <a name="requirements"></a>
- Jax, Optax, Pickle and Matplotlib
