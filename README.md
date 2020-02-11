# Physics-inspired Neural Network (application with NARX for a Thermochemical Energy Storage System)
Read this first for guidelines on using the codes.
This repository contains 4 codes:
  - *PINN_TCES.m*: the main script
  - *plot_best_worst.m*: plotting function
  - *computePHYLOSS.m*: calculation of physical error for loss function
  - *computeJPHY.m*: calculation of physical error Jacobian


# Things to note

  - The training scripts from the MATLAB NN Toolbox is by default located in
    ```sh
    c:/Program Files/MATLAB/'version'/toolbox/nnet/nnet/nntrain
    ```
    It might be a good idea to create a copy of this folder because modifications are necessary. Do not forget to add the location of the copied folder into your MATLAB search path with higher priority.
    
  - The training algorithm used is contained in the script ***trainbr.m***. There are two options: either modify this script directly or create a copy of this script with a different file name.
  
  - In the ***trainbr*** script (or the copy), adjust accordingly the ***initializeTraining***, ***trainingIteration***, and ***computeDX*** functions.
    *NOTE*: The original script was developed for the "MSE+L2" regularization method, and therefore for this method no modification is necessary.

  - In the main script *PINN_TCES.m*, use the name of this modified or copied file (*trainbr* or modified *trainbr*) as a string input to the variable ***trainingFcnString***.


# Modifications of *trainbr* for Regularization Method "MSE"

In all ***initializeTraining***, ***trainingIteration***, and ***computeDX*** functions, set ***<p align="center"><img src="/tex/a849f2cee7244ee19a0e0bd30c97e748.svg?invert_in_darkmode&sanitize=true" align=middle width=40.713337499999994pt height=10.5936072pt/></p>***.


# Modifications of *trainbr* for Regularization Method "MSE+PHY"

  - In the ***initializeTraining*** function, set ***<p align="center"><img src="/tex/a849f2cee7244ee19a0e0bd30c97e748.svg?invert_in_darkmode&sanitize=true" align=middle width=40.713337499999994pt height=10.5936072pt/></p>*** and initialize ***<p align="center"><img src="/tex/39f439719fc36350d276526450d236fa.svg?invert_in_darkmode&sanitize=true" align=middle width=76.20719865pt height=38.332593749999994pt/></p>*** accordingly.
  - In the ***trainingIteration*** function, set ***<p align="center"><img src="/tex/a849f2cee7244ee19a0e0bd30c97e748.svg?invert_in_darkmode&sanitize=true" align=middle width=40.713337499999994pt height=10.5936072pt/></p>***, call the ***computePHYLOSS*** function to calculate physical error to calculate the loss function and call the ***computeJPHY*** function to calculate the approximate Hessian and update the hyperparameters.
  - In the ***computeDX*** function, set ***<p align="center"><img src="/tex/a849f2cee7244ee19a0e0bd30c97e748.svg?invert_in_darkmode&sanitize=true" align=middle width=40.713337499999994pt height=10.5936072pt/></p>***, adjust the calculation of ***num*** to include the approximate Hessian of the physical error, and adjust the calculation of ***den*** to include the multiplication of the Jacobian and the physical errors.


# Modifications of *trainbr* for Regularization Method "MSE+L2+PHY"

  - In the ***initializeTraining*** function, initialize ***<p align="center"><img src="/tex/39f439719fc36350d276526450d236fa.svg?invert_in_darkmode&sanitize=true" align=middle width=76.20719865pt height=38.332593749999994pt/></p>*** accordingly.
  - In the ***trainingIteration*** function, call the ***computePHYLOSS*** function to calculate physical error to calculate the loss function and call the ***computeJPHY*** function to calculate the approximate Hessian and update the hyperparameters.
  - In the ***computeDX*** function, adjust the calculation of ***num*** to include the approximate Hessian of the physical error, and adjust the calculation of ***den*** to include the multiplication of the Jacobian and the physical errors.

