# Hybrid-Modelling-for-Reactor-Model-Discovery-Using-ANN-Classifiers
Overview
This Python codebase is designed for hybrid modeling of chemical reactors, specifically applied to the esterification of benzoic acid. The framework integrates first-principles modeling of reactor kinetics and hydrodynamics with Artificial Neural Networks (ANNs) for automated reactor model classification.

The esterification process is modeled in two types of reactors:
    Continuous Stirred Tank Reactor (CSTR)
    Plug Flow Reactor (PFR)

Each reactor can operate under one of two possible kinetic models, resulting in a total of four physics-based reactor models.
Features
    Physics-Based Modeling:
        Four reactor configurations (CSTR/PFR × Kinetic Model 1/2)
        Includes temperature-dependent reaction kinetics
        Supports both single reactor and CSTR-in-series configurations

    Simulation Engine:
        Generate large datasets over the kinetic parameter space
        Supports solving both CSTR and PFR systems

    Machine Learning Integration:
        Simulated datasets can be used to train ANN classifiers
        Classifiers can predict the most appropriate reactor model given new experimental data

Application Workflow

    Define Kinetic Parameters and Operating Conditions:
        Select kinetic model (0–3)
        Specify inlet concentrations, residence time, and temperature

    Run Physics-Based Simulations:
        Use model_solve() for simulating CSTR or PFR behavior
        Use CSTR_series() for simulating CSTRs in series

    Generate Training Dataset:
        Simulate a wide range of kinetic parameters
        Label data based on the known model type used in simulation

    Train ANN Classifier:
        Input: concentrations, temperature, and possibly other metadata
        Output: predicted model class (0–3)

    Apply to Experimental Data:
        Run the trained classifier on new measurements
        Identify the most probable reactor model

Code Structure
    kinetic_model_: reported in Waldron, C., Pankajakshan, A., Quaglio, M., Cao, E., Galvanin, F., and Gavriilidis, A., 2019. https://doi.org/10.1021/acs.iecr.9b04089![image](https://github.com/user-attachments/assets/3ab6e2c9-8473-4548-81da-391e66694d9d)

    CSTR, PFR: Model equations for different reactor types
    solve_CSTR, solve_PFR: Solvers for steady-state conditions
    CSTR_series: Simulates a chain of CSTRs
    model_solve: Dispatcher function for model simulation

Requirements
    Python 3.x
    NumPy
    SciPy

Usage Example

# Define input variables

x_in = [1.0, 1.0, 0.0, 0.0]  # Initial concentrations

u = [105.0]                 # Temperature in Celsius

theta = [1.0, 0.1, 0.5, 0.3, 0.2, 0.1]  # Kinetic parameters

tau = 100.0                 # Residence time in seconds

km = 3                      # Kinetic model index (0-3)

r = 0                       # Reactor type: 0 for CSTR, 1 for PFR


# Run simulation

x_out = model_solve(tau, x_in, u, theta, r, km)

Citation

If you use this code for your research or publication, please cite the relevant work or acknowledge the original authors.
1)  Quaglio, M., Roberts, L.R., Jaapar, M.S., Fraga, E.S., Dua, V., & Galvanin, F. 2020. https://doi.org/10.1016/j.compchemeng.2020.106759![image](https://github.com/user-attachments/assets/138f4ae3-e244-466b-b26a-612180b54c61)

2)  Sangoi, E., Quaglio, M., Bezzo, F., Galvanin, F., 2024, https://doi.org/10.1016/j.compchemeng.2024.108752.![image](https://github.com/user-attachments/assets/53f3c0fb-0766-4485-b035-ccdd4f272bbe)

3)	Emmanuel Agunloye, Asterios Gavriilidis, Federico Galvanin, Hybrid Modelling Framework for Reactor Model Discovery Using Artificial Neural Networks Classifiers, 1st SUSTENS Meeting – Advances in Sustainable Engineering Systems, a Virtual conference taking place on 4–5 June 2025, accepted for oral Presentation

4)  Emmanuel Agunloye, Asterios Gavriilidis, Federico Galvanin, Reactor Model Discovery Using an Artificial Neural Networks Classifier Framework, ChemEngDayUK2025 Sheffield, University of Sheffield University, accepted for poster presentation.


License

This code is provided for academic and research purposes. For commercial use, please contact the author(s).
