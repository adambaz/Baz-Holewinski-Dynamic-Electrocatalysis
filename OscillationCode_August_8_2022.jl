# Dynamic Electrocatalysis Simulation Code for "Dynamic Electrocatalysis: Examining 
# Resonant Catalytic Rate Enhancement Under Oscillating Electrochemical Potential
# Written by: Adam Baz, Mason Lyons, and Adam Holewinski
# Updated: August 8, 2022

using DifferentialEquations
using NLsolve
using PyCall
using Statistics
using Trapz
using Waveforms
using LaTeXStrings
using TickTock

mpl = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
PyCall.fixqtpath()
np = pyimport("numpy")
mp = pyimport("mpmath")

PyCall.PyDict(mpl["rcParams"])["font.sans-serif"] = ["Arial"] # Makes all plots have Arial font

setprecision(400) # Sets precision for the "bigfloats"

# Model: Sequential three-step mechanism with single poisoning species. 
# Just turn on/off poisoning step by setting Rx(f) pressure to a value, or 0

# Consider the following three-step mechanism: total reaction R <-> P + 3e-
# (1) R(f) + (*) <-> I1(*) + e-
# (2) I1(*) <-> I2(*) + e-
# (3) I2(*) <-> P(f) + (*) + e-
# With a possible faradaic "poisoning" step
# (4) Rx(f) + (*) <-> Ix(*) + e- 

kB = 8.6173e-5 # eV/K
h = 4.1357e-15 # eV*s
T = 298.15 # K
PR = 1 # bar, pressure of reactants 
PP = 1e-3 # bar, pressure of products
PRx = 0 # bar, pressure of poisoning species in fluid-phase
Eref = 0 # V 
Eocv = 1.0 # V vs. reference potential - use to define efficiency with respect to OCP of an overall cell reaction
S0_array_Mref_Eref = [0, -0.0005, -0.001, -0.001, -0.001, -0.0005, 0] # eV/K, [R, TS1, I1, TS2, I2, TS3, P] - standard entropies (relative to reactants) of each species on a reference material at the reference potential
H0_array_Mref_Eref = [0, 0.7, -0.30, 0.55, -0.30, 0.70, 0] # eV, [R, TS1, I1, TS2, I2, TS3, P] - standard enthalpies (relative to reactants) on reference material at reference potential
S0_poisoning_array_Mref_Eref = [0,-0.0005,-0.001] # eV/K, [Rx, TSx, Ix] - standard entropies (relative to poisoning reactant species) on a reference material at a reference potential
H0_poisoning_array_Mref_Eref = [0,0.50,-0.20] # eV, [Rx, TSx, Ix] - standard enthalpies (relative to poisoning reactant species) on a reference material at a reference potential

betas = [0.1,0.9,0.9] # electrochemical symmetry factors
beta_poison = 0.5 # electrochemical symmetry factor for poisoning adsorption step
gamma_BEPs = [0.5,0.5,0.5] # BEP scaling relation slopes
gamma_BEP_poison = 0.5 # BEP scaling relation slope for poisoning adsorption step
del_scaling_params = [0, 1, 0.5, 0] # thermodynamic scaling relation parameters, R, I1, I2, P ... I1 is always "1" since this is the descriptor ... I2 can change, edit it in here ... R and P are always fixed
del_scaling_param_poison = 0.5 # can turn this on if you want the poisoning species to scale with I1*

# This function gets the forward/reverse rate constants and standard enthalpies/free energies of each of the species in the mechanism at a given temperature (T), on a given material (delta_H0_I1), at a given potential (E), given the reference energies, symmetry factors, BEP parameters, and scaling parameters
function getRateConstantsAndEnergies(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison)
    equilibriumConstants = zeros(length(gamma_BEPs)) # equilibrium constant for each elementary step, initialize
    forwardRateConstants = zeros(length(gamma_BEPs)) # forward rate constant for each elementary step, initialize
    backwardRateConstants = zeros(length(gamma_BEPs)) # backward rate constant for each elementary step, initialize
    for i in 1:length(equilibriumConstants) # loop over each elementary step
        equilibriumConstants[i] = exp((S0_array_Mref_Eref[Int((2*i)+1)] - S0_array_Mref_Eref[Int((2*i)-1)])/kB) * exp(-((H0_array_Mref_Eref[Int((2*i)+1)] - H0_array_Mref_Eref[Int((2*i)-1)]) - (E - Eref) + ((del_scaling_params[i+1] - del_scaling_params[i]) * delta_H0_I1))/(kB*T))
        # Make sure these don't go barrierless or less than 0 - enthalpic barriers will be the max of 0, the reaction enthalpy, or the activation enthalpy
        forwardRateConstants[i] = (kB*T/h) * exp((S0_array_Mref_Eref[Int(2*i)] - S0_array_Mref_Eref[Int((2*i)-1)])/kB) * exp(-max(0, ((H0_array_Mref_Eref[Int((2*i)+1)] - H0_array_Mref_Eref[Int((2*i)-1)]) - (E - Eref) + ((del_scaling_params[i+1] - del_scaling_params[i]) * delta_H0_I1)), ((H0_array_Mref_Eref[Int(2*i)] - H0_array_Mref_Eref[Int((2*i)-1)]) - (betas[i] * (E - Eref)) + (gamma_BEPs[i] * ((del_scaling_params[i+1] - del_scaling_params[i]) * delta_H0_I1))))/(kB*T))
        backwardRateConstants[i] = forwardRateConstants[i]/equilibriumConstants[i]
    end
    # Do the poisoning rate constants explicitly here
    equilibriumConstantPoison = exp((S0_poisoning_array_Mref_Eref[3] - S0_poisoning_array_Mref_Eref[1])/kB) * exp(-((H0_poisoning_array_Mref_Eref[3] - H0_poisoning_array_Mref_Eref[1]) - (E - Eref) + (del_scaling_param_poison * delta_H0_I1))/(kB*T))
    forwardRateConstantPoison = (kB*T/h) * exp((S0_poisoning_array_Mref_Eref[2] - S0_poisoning_array_Mref_Eref[1])/kB) * exp(-max(0, ((H0_poisoning_array_Mref_Eref[3] - H0_poisoning_array_Mref_Eref[1]) - (E - Eref) + (del_scaling_param_poison * delta_H0_I1)), ((H0_poisoning_array_Mref_Eref[2] - H0_poisoning_array_Mref_Eref[1]) - (beta_poison * (E - Eref)) + (gamma_BEP_poison * (del_scaling_param_poison * delta_H0_I1))))/(kB*T))
    backwardRateConstantPoison = forwardRateConstantPoison/equilibriumConstantPoison
    # Remember that rateConstants will be all forward constants first, then all backward constants
    forwardRateConstants = cat(forwardRateConstants, forwardRateConstantPoison, dims=(1,1)) # add the poisoning forward rate constant to the array
    backwardRateConstants = cat(backwardRateConstants, backwardRateConstantPoison, dims=(1,1)) # add the poisoning backward rate constant to the array
    rateConstants = cat(forwardRateConstants, backwardRateConstants, dims=(1,1)) # add the backward rate constants onto the end of the forward rate constants vector, and create a new name called rateConstants

    # Get the standard enthalpies and Gibbs free energies on a given material / at a given potential for the energy diagrams
    H0_array_for_energy_diagram = zeros(length(H0_array_Mref_Eref))
    G0_array_for_energy_diagram = zeros(length(H0_array_Mref_Eref))
    for i in 2:length(H0_array_Mref_Eref) # loop over each species in the mechanism. Start on index "2" since first element is always 0 (i.e reactant species)
        if mod(i,2) == 0 # if the index is even, this is a transition state species
            H0_array_for_energy_diagram[i] = H0_array_for_energy_diagram[i-1] + max(0, ((H0_array_Mref_Eref[i+1] - H0_array_Mref_Eref[i-1]) - (E - Eref) + ((del_scaling_params[Int((i/2)+1)] - del_scaling_params[Int(i/2)]) * delta_H0_I1)), ((H0_array_Mref_Eref[i] - H0_array_Mref_Eref[i-1]) - (betas[Int(i/2)] * (E - Eref)) + (gamma_BEPs[Int(i/2)] * ((del_scaling_params[Int((i/2)+1)] - del_scaling_params[Int(i/2)]) * delta_H0_I1))))
            G0_array_for_energy_diagram[i] = H0_array_for_energy_diagram[i] - T*(S0_array_Mref_Eref[i])
        elseif mod(i,2) != 0 # if the index is odd, this is an intermediate (or product) species
            H0_array_for_energy_diagram[i] = H0_array_for_energy_diagram[i-2] + ((H0_array_Mref_Eref[i] - H0_array_Mref_Eref[i-2]) - (E - Eref) + ((del_scaling_params[Int((i+1)/2)] - del_scaling_params[Int((i-1)/2)]) * delta_H0_I1))
            G0_array_for_energy_diagram[i] = H0_array_for_energy_diagram[i] - T*(S0_array_Mref_Eref[i])
        end
    end

    # Try to calculate an "apparent rate constant" for situation where I1* is MARI but TS3 has a high DRC, such that there is an "apparent barrier" between I1 and TS3.
    # Can also do one for the apparent barrier between R and TS2, and R and TS3
    # Could be more scenarios for a larger mechanism but for the three step, these are the only ones to consider.
    # Easiest to just use the potential/material dependent standard free energies that we computed for the FED above as the inputs here, instead of explicitly defining like we did the elementary rate constants.

    k_app_R_TS2 = (kB*T/h) * exp(-(G0_array_for_energy_diagram[4] - G0_array_for_energy_diagram[1])/(kB*T))
    k_app_R_TS3 = (kB*T/h) * exp(-(G0_array_for_energy_diagram[6] - G0_array_for_energy_diagram[1])/(kB*T))
    k_app_I1_TS3 = (kB*T/h) * exp(-(G0_array_for_energy_diagram[6] - G0_array_for_energy_diagram[3])/(kB*T))

    # Need to make a separate array of enthalpies and free energies for the energy diagram corresponding to the poisoning species, just do this explicitly
    H0_array_for_energy_diagram_poison = [H0_poisoning_array_Mref_Eref[1], max(H0_poisoning_array_Mref_Eref[1], ((H0_poisoning_array_Mref_Eref[3]) - (E - Eref) + (del_scaling_param_poison * delta_H0_I1)), ((H0_poisoning_array_Mref_Eref[2]) - (beta_poison * (E - Eref)) + (gamma_BEP_poison * (del_scaling_param_poison * delta_H0_I1)))), ((H0_poisoning_array_Mref_Eref[3]) - (E - Eref) + (del_scaling_param_poison * delta_H0_I1))]
    G0_array_for_energy_diagram_poison = H0_array_for_energy_diagram_poison .- (T .* S0_poisoning_array_Mref_Eref)

    return rateConstants, H0_array_for_energy_diagram, G0_array_for_energy_diagram, H0_array_for_energy_diagram_poison, G0_array_for_energy_diagram_poison, k_app_R_TS2, k_app_R_TS3, k_app_I1_TS3
end

# Makes an energy diagram (Gibbs or enthalpy).
function makeEnergyDiagram(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, type, color, plotMethod, axs) # need to include "axs" object for the subplots to work
    output = getRateConstantsAndEnergies(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison)
    enthalpies = output[2]
    Gibbs = output[3]
    enthalpies_poison = output[4] # add the poisoning species here
    Gibbs_poison = output[5] # add the poisoning species here
    if type == "enthalpy"
        energies = enthalpies
        energies_poison = enthalpies_poison
    elseif type == "Gibbs"
        energies = Gibbs
        energies_poison = Gibbs_poison
    end
    count_odd = 0
    count_even = 0
    color_poison = "r" # just always make the poison color red, no need to change the function inputs
    if plotMethod == "independent"
        LineWidthVal = 5 # thick line for the first point
    elseif plotMethod == "runFullAnalysis"
        LineWidthVal = 4 # make less thick for subplot
    end
    LineProperties = string(color,"-") # this will concatenate the color from the function input with the linestype, i.e. "k-"
    LinePropertiesPoison = string(color_poison,"-")
    if plotMethod == "independent"
        plt.figure(figsize=(12,12)) # only want to instantiate a figure inside this function if we plan on plotting by just calling this function alone
    end
    for i in 1:(2*length(energies)-1)
        if mod(i,2) != 0 && i != 1 # if "i" is odd and not equal to 1 (don't start adding on the first index)
            count_odd += 1
            if plotMethod == "independent"
                LineWidthVal = 4
            elseif plotMethod == "runFullAnalysis"
                LineWidthVal = 4
            end
            LineProperties = string(color,"-")
            LinePropertiesPoison = string(color_poison,"-")
        elseif mod(i,2) == 0 # if "i" is even
            count_even += 1
            if plotMethod == "independent"
                LineWidthVal = 2
            elseif plotMethod == "runFullAnalysis"
                LineWidthVal = 2
            end
            LineProperties = string(color,":")
            LinePropertiesPoison = string(color_poison,":")
        end
        if plotMethod == "independent" || plotMethod == "vary" # in either of these cases, plot as you normally would.
            plt.plot([i,i+1],[energies[1+count_odd],energies[1+count_even]],LineProperties,linewidth=LineWidthVal)
            plt.xlabel("Reaction Coordinate",fontsize = 16)
            if type == "enthalpy"
                plt.ylabel("Enthalpy (eV)",fontsize=24)
            elseif type == "Gibbs"
                plt.ylabel("Free Energy (eV)",fontsize = 16)
            end
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
        elseif plotMethod == "runFullAnalysis" # for this case, we need to use the "axs" object so that the plots that get made in here get added into the larger subplot from the "runFullAnalysis" function
            axs[1,1].plot([i,i+1],[energies[1+count_odd],energies[1+count_even]],LineProperties,linewidth=LineWidthVal)
            if i <= 5 && PRx != 0 # there are only 3 species for the poison so only tell to plot if we are less than 3 in the species count
                axs[1,1].plot([i,i+1],[energies_poison[1+count_odd],energies_poison[1+count_even]],LinePropertiesPoison,linewidth=LineWidthVal)
            end
            axs[1,1].set_xlabel("Reaction Coordinate", labelpad = 5, fontsize = 18)
            axs[1,1].set_ylabel("Free Energy (eV)", fontsize = 18, labelpad = 10)
            axs[1,1].set_title(" R   TS1   I1   TS2   I2   TS3   P", fontsize = 16)
            axs[1,1].set_xticks([])
            axs[1,1].set_xticklabels([]) # set as empty to hide
            axs[1,1].tick_params(axis="y",labelsize=16)
            axs[1,1].text(-1.25,1.03,"A", fontsize=18, fontweight="bold")

        end

    end
    if plotMethod == "independent" # again only show plot in this function if we want it to be independent. Otherwise it will show when "show()" is called in the other functions calling this function
        plt.show()
    end
end

function transientRateFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, integrationTime, initialCondition)

    function DiffyQ(du,u,p,t)

        t_I1 = u[1] # coverage of I1*
        t_I2 = u[2] # coverage of I2*
        t_Ix = u[3] # coverage of poisoning species Ix*
        t_star = 1 - u[1] - u[2] - u[3] # free sites

        rateConstantFunctionOutput = getRateConstantsAndEnergies(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison)
        rateConstants = rateConstantFunctionOutput[1] # rate constants are [k1f,k2f,k3f,kPf,k1b,k2b,k3b,kPb], kPf and kPb are the forward and backward rate constants for the poisoning step

        du[1] = (rateConstants[1] * PR * t_star) - (rateConstants[5] * t_I1) - (rateConstants[2] * t_I1) + (rateConstants[6] * t_I2)
        du[2] = (rateConstants[2] * t_I1) - (rateConstants[6] * t_I2) - (rateConstants[3] * t_I2) + (rateConstants[7] * PP * t_star)
        du[3] = (rateConstants[4] * PRx * t_star) - (rateConstants[8] * t_Ix)

    end

    timeSpan = (0.0, integrationTime)
    problem = ODEProblem(DiffyQ, big.(initialCondition), big.(timeSpan)) # use this if need the bigFloats for improved precision
    solution = solve(problem, RadauIIA5(), reltol = 1e-8, abstol = 1e-8, dense=true)
    # Try to access the actual time points from the solution
    timeArray = solution.t
    arraySolution = convert(Array, solution)
    t_I1 = arraySolution[1,:]
    t_I2 = arraySolution[2,:]
    t_Ix = arraySolution[3,:]
    t_star = 1 .- t_I1 .- t_I2 .- t_Ix
    # Now that we have the "solved" coverages, we need to get the rate constants again to get the "rates" (the rate constants inside the "DiffyQ" function remain inside that scope)
    rateConstantFunctionOutput = getRateConstantsAndEnergies(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison)
    rateConstants = rateConstantFunctionOutput[1] # rate constants are [k1f,k2f,k3f,kPf,k1b,k2b,k3b,kPb], kPf and kPb are the forward and backward rate constants for the poisoning step

    # Get the net forward rates of each elementary step. These are "arrays" so need to use "." notation for each operation
    # At steady-state, r1 = r2 = r3. For transient operation, this is not necessarily the case.
    # Remember # rate constants are [k1f,k2f,k3f,kPf,k1b,k2b,k3b,kPb], kPf and kPb are the forward and backward rate constants for the poisoning step
    r1 = (rateConstants[1] .* PR .* t_star) .- (rateConstants[5] .* t_I1)
    r2 = (rateConstants[2] .* t_I1) .- (rateConstants[6] .* t_I2)
    r3 = (rateConstants[3] .* t_I2) .- (rateConstants[7] .* PP .* t_star)

    transientRate_P = r3
    transientRate_electron_tot = r1 .+ r2 .+ r3
    transientRate_electron_r1 = r1
    transientRate_electron_r2 = r2
    transientRate_electron_r3 = r3

    return transientRate_P, transientRate_electron_tot, transientRate_electron_r1, transientRate_electron_r2, transientRate_electron_r3, t_I1, t_I2, t_Ix, t_star, timeArray
end

function steadyStateRateFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)

    function f!(fvec,x) # uses a slightly different notation for nlsolve

        t_I1 = x[1] # coverage of I1
        t_I2 = x[2] # coverage of I2
        t_Ix = x[3] # coverage of Ix
        t_star = 1 - t_I1 - t_I2 - t_Ix # free sites

        rateConstantFunctionOutput = getRateConstantsAndEnergies(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison)
        rateConstants = rateConstantFunctionOutput[1] # rate constants are [k1f,k2f,k3f,kPf,k1b,k2b,k3b,kPb], kPf and kPb are the forward and backward rate constants for the poisoning step

        fvec[1] = (rateConstants[1] * PR * t_star) - (rateConstants[5] * t_I1) - (rateConstants[2] * t_I1) + (rateConstants[6] * t_I2)
        fvec[2] = (rateConstants[2] * t_I1) - (rateConstants[6] * t_I2) - (rateConstants[3] * t_I2) + (rateConstants[7] * PP * t_star)
        fvec[3] = (rateConstants[4] * PRx * t_star) - (rateConstants[8] * t_Ix)

    end

    res = nlsolve(f!, big.(initialCondition), ftol = 1e-60, method = :newton) # can always keep the bigFloat precision on this since these need higher tolerances to converage, and also aren't very slow
    solution = res.zero # solves using a root-finding algorithm
    arraySolution = convert(Array,solution) # converts the solution object into an array to work with
    t_I1 = arraySolution[1,:]
    t_I2 = arraySolution[2,:]
    t_Ix = arraySolution[3,:]
    t_star = 1 .- t_I1 .- t_I2 .- t_Ix
    SS_I1 = last(t_I1) # steady-state solution coverage of I1 - just a single value
    SS_I2 = last(t_I2) # steady-state solution coverage of I2 - just a single value
    SS_Ix = last(t_Ix) # steady-state solution coverage of Ix - just a single value
    SS_star = last(t_star) # steady-state solution coverage of free sites - just a single value - don't really need this though

    # Now that we have the "solved" coverages, we need to get the rate constants again to get the "rates" (the rate constants inside the "f!" function remain inside that scope)
    rateConstantFunctionOutput = getRateConstantsAndEnergies(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison)
    rateConstants = rateConstantFunctionOutput[1] # rate constants are [k1f,k2f,k3f,kPf,k1b,k2b,k3b,kPb], kPf and kPb are the forward and backward rate constants for the poisoning step
    k_app_R_TS2 = rateConstantFunctionOutput[6]
    k_app_R_TS3 = rateConstantFunctionOutput[7]
    k_app_I1_TS3 = rateConstantFunctionOutput[8]

    # At steady-state, r1 = r2 = r3. For transient operation, this is not necessarily the case.
    r1 = (rateConstants[1] .* PR .* SS_star) .- (rateConstants[5] .* SS_I1)
    r2 = (rateConstants[2] .* SS_I1) .- (rateConstants[6] .* SS_I2)
    r3 = (rateConstants[3] .* SS_I2) .- (rateConstants[7] .* PP .* SS_star)

    if productType == "electron"
        steadyStateRate = r1 .+ r2 .+ r3
    elseif productType == "P"
        steadyStateRate = r3
    end

    # Also output the forward rate constants to draw extrapolated "maximum rate" lines on the volcano plots
    k1f = rateConstants[1]
    k2f = rateConstants[2]
    k3f = rateConstants[3]

    return steadyStateRate, SS_I1, SS_I2, SS_Ix, SS_star, k1f, k2f, k3f, k_app_R_TS2, k_app_R_TS3, k_app_I1_TS3
end

# This function will calculate the steady-state rates, coverages, DRCs, and more if desired (rate constants, standard Gibbs energies) over a set of binding energies specified in "delta_H0_I1_array" at a constant potential, E.
function steadyStateRateFunctionVaryBindingEnergy(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array, T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, plotMethod, productType)

    # Initialize arrays to place the "solved" steady-state rates and coverages at each binding energy condition
    steadyStateRateArray = zeros(length(delta_H0_I1_array))
    steadyStateI1CoverageArray = zeros(length(delta_H0_I1_array))
    steadyStateI2CoverageArray = zeros(length(delta_H0_I1_array))
    steadyStateIxCoverageArray = zeros(length(delta_H0_I1_array))
    DRC_TS1_array = zeros(length(delta_H0_I1_array))
    DRC_I1_array = zeros(length(delta_H0_I1_array))
    DRC_TS2_array = zeros(length(delta_H0_I1_array))
    DRC_I2_array = zeros(length(delta_H0_I1_array))
    DRC_TS3_array = zeros(length(delta_H0_I1_array))
    DRC_TSx_array = zeros(length(delta_H0_I1_array))
    DRC_Ix_array = zeros(length(delta_H0_I1_array))
    # Get forward rate constants
    k1f_array = zeros(length(delta_H0_I1_array))
    k2f_array = zeros(length(delta_H0_I1_array))
    k3f_array = zeros(length(delta_H0_I1_array))
    # Try the apparent rate constants
    k_app_R_TS2_array = zeros(length(delta_H0_I1_array))
    k_app_R_TS3_array = zeros(length(delta_H0_I1_array))
    k_app_I1_TS3_array = zeros(length(delta_H0_I1_array))

    for i in 1:length(delta_H0_I1_array)
        steadyStateRateFunctionOutput = steadyStateRateFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[i], T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)
        steadyStateRateArray[i] = steadyStateRateFunctionOutput[1]
        steadyStateI1CoverageArray[i] = steadyStateRateFunctionOutput[2]
        steadyStateI2CoverageArray[i] = steadyStateRateFunctionOutput[3]
        steadyStateIxCoverageArray[i] = steadyStateRateFunctionOutput[4]
        k1f_array[i] = steadyStateRateFunctionOutput[6]
        k2f_array[i] = steadyStateRateFunctionOutput[7]
        k3f_array[i] = steadyStateRateFunctionOutput[8]
        k_app_R_TS2_array[i] = steadyStateRateFunctionOutput[9]
        k_app_R_TS3_array[i] = steadyStateRateFunctionOutput[10]
        k_app_I1_TS3_array[i] = steadyStateRateFunctionOutput[11]

        # Let's try to add degree of rate control analysis here. Perturb the standard Gibbs free energy of each species (excluding reactants). This is the same as perturbing the standard enthalpy, which is inside the "H0_array_Mref_Eref" array
        del_G0_DRC = -0.000001 # perturbation to Gibbs free energy for DRC analysis
        # Perturb TS1 the H0 arrays
        H0_array_Mref_Eref_perturb_TS1 = [H0_array_Mref_Eref[1], H0_array_Mref_Eref[2] + del_G0_DRC, H0_array_Mref_Eref[3], H0_array_Mref_Eref[4], H0_array_Mref_Eref[5], H0_array_Mref_Eref[6], H0_array_Mref_Eref[7]]
        H0_array_Mref_Eref_perturb_I1 = [H0_array_Mref_Eref[1], H0_array_Mref_Eref[2], H0_array_Mref_Eref[3] + del_G0_DRC, H0_array_Mref_Eref[4], H0_array_Mref_Eref[5], H0_array_Mref_Eref[6], H0_array_Mref_Eref[7]]
        H0_array_Mref_Eref_perturb_TS2 = [H0_array_Mref_Eref[1], H0_array_Mref_Eref[2], H0_array_Mref_Eref[3], H0_array_Mref_Eref[4] + del_G0_DRC, H0_array_Mref_Eref[5], H0_array_Mref_Eref[6], H0_array_Mref_Eref[7]]
        H0_array_Mref_Eref_perturb_I2 = [H0_array_Mref_Eref[1], H0_array_Mref_Eref[2], H0_array_Mref_Eref[3], H0_array_Mref_Eref[4], H0_array_Mref_Eref[5] + del_G0_DRC, H0_array_Mref_Eref[6], H0_array_Mref_Eref[7]]
        H0_array_Mref_Eref_perturb_TS3 = [H0_array_Mref_Eref[1], H0_array_Mref_Eref[2], H0_array_Mref_Eref[3], H0_array_Mref_Eref[4], H0_array_Mref_Eref[5], H0_array_Mref_Eref[6] + del_G0_DRC, H0_array_Mref_Eref[7]]
        H0_poisoning_array_Mref_Eref_perturb_TSx = [H0_poisoning_array_Mref_Eref[1], H0_poisoning_array_Mref_Eref[2] + del_G0_DRC, H0_poisoning_array_Mref_Eref[3]]
        H0_poisoning_array_Mref_Eref_perturb_Ix = [H0_poisoning_array_Mref_Eref[1], H0_poisoning_array_Mref_Eref[2], H0_poisoning_array_Mref_Eref[3] + del_G0_DRC]

        # Get a perturbed steady-state rate for each of the perturbed enthalpy arrays
        # steadyStateRateFunction input is: steadyStateRateFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[i], T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType
        steadyStateRateFunctionOutputPerturbTS1 = steadyStateRateFunction(H0_array_Mref_Eref_perturb_TS1, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[i], T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)
        steadyStateRatePerturbTS1 = steadyStateRateFunctionOutputPerturbTS1[1]
        steadyStateRateFunctionOutputPerturbI1 = steadyStateRateFunction(H0_array_Mref_Eref_perturb_I1, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[i], T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)
        steadyStateRatePerturbI1 = steadyStateRateFunctionOutputPerturbI1[1]
        steadyStateRateFunctionOutputPerturbTS2 = steadyStateRateFunction(H0_array_Mref_Eref_perturb_TS2, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[i], T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)
        steadyStateRatePerturbTS2 = steadyStateRateFunctionOutputPerturbTS2[1]
        steadyStateRateFunctionOutputPerturbI2 = steadyStateRateFunction(H0_array_Mref_Eref_perturb_I2, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[i], T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)
        steadyStateRatePerturbI2 = steadyStateRateFunctionOutputPerturbI2[1]
        steadyStateRateFunctionOutputPerturbTS3 = steadyStateRateFunction(H0_array_Mref_Eref_perturb_TS3, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[i], T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)
        steadyStateRatePerturbTS3 = steadyStateRateFunctionOutputPerturbTS3[1]
        steadyStateRateFunctionOutputPerturbTSx = steadyStateRateFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref_perturb_TSx, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[i], T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)
        steadyStateRatePerturbTSx = steadyStateRateFunctionOutputPerturbTSx[1]
        steadyStateRateFunctionOutputPerturbIx = steadyStateRateFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref_perturb_Ix, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[i], T, E, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)
        steadyStateRatePerturbIx = steadyStateRateFunctionOutputPerturbIx[1]

        # Get the DRCs
        DRC_TS1_array[i] = (log(steadyStateRatePerturbTS1) - log(steadyStateRateArray[i])) / (-del_G0_DRC/(kB*T))
        DRC_I1_array[i] = (log(steadyStateRatePerturbI1) - log(steadyStateRateArray[i])) / (-del_G0_DRC/(kB*T))
        DRC_TS2_array[i] = (log(steadyStateRatePerturbTS2) - log(steadyStateRateArray[i])) / (-del_G0_DRC/(kB*T))
        DRC_I2_array[i] = (log(steadyStateRatePerturbI2) - log(steadyStateRateArray[i])) / (-del_G0_DRC/(kB*T))
        DRC_TS3_array[i] = (log(steadyStateRatePerturbTS3) - log(steadyStateRateArray[i])) / (-del_G0_DRC/(kB*T))
        DRC_TSx_array[i] = (log(steadyStateRatePerturbTSx) - log(steadyStateRateArray[i])) / (-del_G0_DRC/(kB*T))
        DRC_Ix_array[i] = (log(steadyStateRatePerturbIx) - log(steadyStateRateArray[i])) / (-del_G0_DRC/(kB*T))

    end

    # Get the maximum steady-state rate over the given binding energy interval for comparing to later on (in calculating enhancement factors)
    maxSteadyStateRate = maximum(steadyStateRateArray)

    return steadyStateRateArray, steadyStateI1CoverageArray, steadyStateI2CoverageArray, steadyStateIxCoverageArray, maxSteadyStateRate, DRC_TS1_array, DRC_I1_array, DRC_TS2_array, DRC_I2_array, DRC_TS3_array, DRC_TSx_array, DRC_Ix_array, k1f_array, k2f_array, k3f_array, k_app_R_TS2_array, k_app_R_TS3_array, k_app_I1_TS3_array
end

# This function will calculate the steady-state rates, coverages, DRCs, etc over a set of potentials specified in "E_array" at a constant descriptor binding energy, delta_H0_I1.
function steadyStateRateFunctionVaryPotential(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E_array, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, plotMethod, productType)

    # Initialize arrays to place the "solved" steady-state rates and coverages at each potential condition
    steadyStateRateArray = zeros(length(E_array))
    steadyStateI1CoverageArray = zeros(length(E_array))
    steadyStateI2CoverageArray = zeros(length(E_array))
    steadyStateIxCoverageArray = zeros(length(E_array))
    DRC_TS1_array = zeros(length(E_array))
    DRC_I1_array = zeros(length(E_array))
    DRC_TS2_array = zeros(length(E_array))
    DRC_I2_array = zeros(length(E_array))
    DRC_TS3_array = zeros(length(E_array))
    DRC_TSx_array = zeros(length(E_array))
    DRC_Ix_array = zeros(length(E_array))
    k1f_array = zeros(length(E_array))
    k2f_array = zeros(length(E_array))
    k3f_array = zeros(length(E_array))
    # Try the apparent rate constants
    k_app_R_TS2_array = zeros(length(E_array))
    k_app_R_TS3_array = zeros(length(E_array))
    k_app_I1_TS3_array = zeros(length(E_array))

    for i in 1:length(E_array)
        steadyStateRateFunctionOutput = steadyStateRateFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E_array[i], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)
        steadyStateRateArray[i] = steadyStateRateFunctionOutput[1]
        steadyStateI1CoverageArray[i] = steadyStateRateFunctionOutput[2]
        steadyStateI2CoverageArray[i] = steadyStateRateFunctionOutput[3]
        steadyStateIxCoverageArray[i] = steadyStateRateFunctionOutput[4]
        k1f_array[i] = steadyStateRateFunctionOutput[6]
        k2f_array[i] = steadyStateRateFunctionOutput[7]
        k3f_array[i] = steadyStateRateFunctionOutput[8]
        k_app_R_TS2_array[i] = steadyStateRateFunctionOutput[9]
        k_app_R_TS3_array[i] = steadyStateRateFunctionOutput[10]
        k_app_I1_TS3_array[i] = steadyStateRateFunctionOutput[11]

        # Let's try to add degree of rate control analysis here. Perturb the standard Gibbs free energy of each species (excluding reactants). This is the same as perturbing the standard enthalpy, which is inside the "H0_array_Mref_Eref" array
        del_G0_DRC = -0.000001 # perturbation to Gibbs free energy for DRC analysis
        # Perturb TS1 the H0 arrays
        H0_array_Mref_Eref_perturb_TS1 = [H0_array_Mref_Eref[1], H0_array_Mref_Eref[2] + del_G0_DRC, H0_array_Mref_Eref[3], H0_array_Mref_Eref[4], H0_array_Mref_Eref[5], H0_array_Mref_Eref[6], H0_array_Mref_Eref[7]]
        H0_array_Mref_Eref_perturb_I1 = [H0_array_Mref_Eref[1], H0_array_Mref_Eref[2], H0_array_Mref_Eref[3] + del_G0_DRC, H0_array_Mref_Eref[4], H0_array_Mref_Eref[5], H0_array_Mref_Eref[6], H0_array_Mref_Eref[7]]
        H0_array_Mref_Eref_perturb_TS2 = [H0_array_Mref_Eref[1], H0_array_Mref_Eref[2], H0_array_Mref_Eref[3], H0_array_Mref_Eref[4] + del_G0_DRC, H0_array_Mref_Eref[5], H0_array_Mref_Eref[6], H0_array_Mref_Eref[7]]
        H0_array_Mref_Eref_perturb_I2 = [H0_array_Mref_Eref[1], H0_array_Mref_Eref[2], H0_array_Mref_Eref[3], H0_array_Mref_Eref[4], H0_array_Mref_Eref[5] + del_G0_DRC, H0_array_Mref_Eref[6], H0_array_Mref_Eref[7]]
        H0_array_Mref_Eref_perturb_TS3 = [H0_array_Mref_Eref[1], H0_array_Mref_Eref[2], H0_array_Mref_Eref[3], H0_array_Mref_Eref[4], H0_array_Mref_Eref[5], H0_array_Mref_Eref[6] + del_G0_DRC, H0_array_Mref_Eref[7]]
        H0_poisoning_array_Mref_Eref_perturb_TSx = [H0_poisoning_array_Mref_Eref[1], H0_poisoning_array_Mref_Eref[2] + del_G0_DRC, H0_poisoning_array_Mref_Eref[3]]
        H0_poisoning_array_Mref_Eref_perturb_Ix = [H0_poisoning_array_Mref_Eref[1], H0_poisoning_array_Mref_Eref[2], H0_poisoning_array_Mref_Eref[3] + del_G0_DRC]

        # Get a perturbed steady-state rate for each of the perturbed enthalpy arrays
        steadyStateRateFunctionOutputPerturbTS1 = steadyStateRateFunction(H0_array_Mref_Eref_perturb_TS1, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E_array[i], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)
        steadyStateRatePerturbTS1 = steadyStateRateFunctionOutputPerturbTS1[1]
        steadyStateRateFunctionOutputPerturbI1 = steadyStateRateFunction(H0_array_Mref_Eref_perturb_I1, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E_array[i], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)
        steadyStateRatePerturbI1 = steadyStateRateFunctionOutputPerturbI1[1]
        steadyStateRateFunctionOutputPerturbTS2 = steadyStateRateFunction(H0_array_Mref_Eref_perturb_TS2, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E_array[i], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)
        steadyStateRatePerturbTS2 = steadyStateRateFunctionOutputPerturbTS2[1]
        steadyStateRateFunctionOutputPerturbI2 = steadyStateRateFunction(H0_array_Mref_Eref_perturb_I2, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E_array[i], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)
        steadyStateRatePerturbI2 = steadyStateRateFunctionOutputPerturbI2[1]
        steadyStateRateFunctionOutputPerturbTS3 = steadyStateRateFunction(H0_array_Mref_Eref_perturb_TS3, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E_array[i], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)
        steadyStateRatePerturbTS3 = steadyStateRateFunctionOutputPerturbTS3[1]
        steadyStateRateFunctionOutputPerturbTSx = steadyStateRateFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref_perturb_TSx, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E_array[i], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)
        steadyStateRatePerturbTSx = steadyStateRateFunctionOutputPerturbTSx[1]
        steadyStateRateFunctionOutputPerturbIx = steadyStateRateFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref_perturb_Ix, S0_poisoning_array_Mref_Eref, delta_H0_I1, T, E_array[i], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, productType)
        steadyStateRatePerturbIx = steadyStateRateFunctionOutputPerturbIx[1]

        # Get the DRCs
        DRC_TS1_array[i] = (log(steadyStateRatePerturbTS1) - log(steadyStateRateArray[i])) / (-del_G0_DRC/(kB*T))
        DRC_I1_array[i] = (log(steadyStateRatePerturbI1) - log(steadyStateRateArray[i])) / (-del_G0_DRC/(kB*T))
        DRC_TS2_array[i] = (log(steadyStateRatePerturbTS2) - log(steadyStateRateArray[i])) / (-del_G0_DRC/(kB*T))
        DRC_I2_array[i] = (log(steadyStateRatePerturbI2) - log(steadyStateRateArray[i])) / (-del_G0_DRC/(kB*T))
        DRC_TS3_array[i] = (log(steadyStateRatePerturbTS3) - log(steadyStateRateArray[i])) / (-del_G0_DRC/(kB*T))
        DRC_TSx_array[i] = (log(steadyStateRatePerturbTSx) - log(steadyStateRateArray[i])) / (-del_G0_DRC/(kB*T))
        DRC_Ix_array[i] = (log(steadyStateRatePerturbIx) - log(steadyStateRateArray[i])) / (-del_G0_DRC/(kB*T))

    end

    # Get the maximum steady-state rate over the potential interval for calculating enhancement factors later on.
    maxSteadyStateRate = maximum(steadyStateRateArray)

    return steadyStateRateArray, steadyStateI1CoverageArray, steadyStateI2CoverageArray, steadyStateIxCoverageArray, maxSteadyStateRate, DRC_TS1_array, DRC_I1_array, DRC_TS2_array, DRC_I2_array, DRC_TS3_array, DRC_TSx_array, DRC_Ix_array, k1f_array, k2f_array, k3f_array, k_app_R_TS2_array, k_app_R_TS3_array, k_app_I1_TS3_array
end

# This function makes a 2D heat map of rates, coverages, and DRCs over binding energies (delta_H0_I1_array) and potentials (E_array)
function makeTwoDimensionalPotentialAndBindingEnergyVolcano(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array, T, E_array, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, plotMethod, productType)

    steadyStateRateMatrix = zeros(length(delta_H0_I1_array),length(E_array))
    steadyStateI1CoverageMatrix = zeros(length(delta_H0_I1_array),length(E_array))
    steadyStateI2CoverageMatrix = zeros(length(delta_H0_I1_array),length(E_array))
    steadyStateIxCoverageMatrix = zeros(length(delta_H0_I1_array),length(E_array))
    DRC_TS1_matrix = zeros(length(delta_H0_I1_array),length(E_array))
    DRC_I1_matrix = zeros(length(delta_H0_I1_array),length(E_array))
    DRC_TS2_matrix = zeros(length(delta_H0_I1_array),length(E_array))
    DRC_I2_matrix = zeros(length(delta_H0_I1_array),length(E_array))
    DRC_TS3_matrix = zeros(length(delta_H0_I1_array),length(E_array))
    DRC_TSx_matrix = zeros(length(delta_H0_I1_array),length(E_array))
    DRC_Ix_matrix = zeros(length(delta_H0_I1_array),length(E_array))
    k1f_matrix = zeros(length(delta_H0_I1_array),length(E_array))
    k2f_matrix = zeros(length(delta_H0_I1_array),length(E_array))
    k3f_matrix = zeros(length(delta_H0_I1_array),length(E_array))
    # Try the apparent rate constants
    k_app_R_TS2_matrix = zeros(length(delta_H0_I1_array),length(E_array))
    k_app_R_TS3_matrix = zeros(length(delta_H0_I1_array),length(E_array))
    k_app_I1_TS3_matrix = zeros(length(delta_H0_I1_array),length(E_array))
    # And the max rates
    maxSteadyStateRate_array = zeros(length(E_array))

    for i in 1:length(E_array)
        steadyStateRateFunctionVaryBindingEnergyOutput = steadyStateRateFunctionVaryBindingEnergy(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array, T, E_array[i], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, plotMethod, productType)
        steadyStateRateMatrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[1]
        steadyStateI1CoverageMatrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[2]
        steadyStateI2CoverageMatrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[3]
        steadyStateIxCoverageMatrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[4]
        DRC_TS1_matrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[6]
        DRC_I1_matrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[7]
        DRC_TS2_matrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[8]
        DRC_I2_matrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[9]
        DRC_TS3_matrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[10]
        DRC_TSx_matrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[11]
        DRC_Ix_matrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[12]
        k1f_matrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[13]
        k2f_matrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[14]
        k3f_matrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[15]
        k_app_R_TS2_matrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[16]
        k_app_R_TS3_matrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[17]
        k_app_I1_TS3_matrix[:,i] = steadyStateRateFunctionVaryBindingEnergyOutput[18]
        # And the max rates
        maxSteadyStateRate_array[i] = steadyStateRateFunctionVaryBindingEnergyOutput[5]
    end

    # Decided that I wanted to plot binding energy on x axis and potential on y axis, so just transpose the final matrices so I don't need to change the loop structure above
    steadyStateRateMatrix = transpose(steadyStateRateMatrix)
    steadyStateI1CoverageMatrix = transpose(steadyStateI1CoverageMatrix)
    steadyStateI2CoverageMatrix = transpose(steadyStateI2CoverageMatrix)
    steadyStateIxCoverageMatrix = transpose(steadyStateIxCoverageMatrix)
    DRC_TS1_matrix = transpose(DRC_TS1_matrix)
    DRC_I1_matrix = transpose(DRC_I1_matrix)
    DRC_TS2_matrix = transpose(DRC_TS2_matrix)
    DRC_I2_matrix = transpose(DRC_I2_matrix)
    DRC_TS3_matrix = transpose(DRC_TS3_matrix)
    DRC_TSx_matrix = transpose(DRC_TSx_matrix)
    DRC_Ix_matrix = transpose(DRC_Ix_matrix)
    k1f_matrix = transpose(k1f_matrix)
    k2f_matrix = transpose(k2f_matrix)
    k3f_matrix = transpose(k3f_matrix)
    k_app_R_TS2_matrix = transpose(k_app_R_TS2_matrix)
    k_app_R_TS3_matrix = transpose(k_app_R_TS3_matrix)
    k_app_I1_TS3_matrix = transpose(k_app_I1_TS3_matrix)
    maxSteadyStateRate_array = transpose(maxSteadyStateRate_array)
    

    return steadyStateRateMatrix, steadyStateI1CoverageMatrix, steadyStateI2CoverageMatrix, steadyStateIxCoverageMatrix, DRC_TS1_matrix, DRC_I1_matrix, DRC_TS2_matrix, DRC_I2_matrix, DRC_TS3_matrix, DRC_TSx_matrix, DRC_Ix_matrix, k1f_matrix, k2f_matrix, k3f_matrix, k_app_R_TS2_matrix, k_app_R_TS3_matrix, k_app_I1_TS3_matrix, maxSteadyStateRate_array
end

# This generates the 3 dimensional TOC figure of the TOF volcano with some extrapolated "maximum rate" lines drawn on - just call this function independently to generate the figure, not included in the "runFullAnalysis" function.
# You may need to play around when deciding which extrapolated "k" to overlay on the volcano - changing the conditions can change which one is "limiting" on either end, but the 2D plots should help elucidate which is which.
function make3Dfigure(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array, T, E_array, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, plotMethod, productType)

    twoDvolcanoOutput = makeTwoDimensionalPotentialAndBindingEnergyVolcano(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array, T, E_array, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, plotMethod, productType)
    steadyStateRateMatrix = twoDvolcanoOutput[1]
    steadyStateI1CoverageMatrix = twoDvolcanoOutput[2]
    steadyStateI2CoverageMatrix = twoDvolcanoOutput[3]
    steadyStateIxCoverageMatrix = twoDvolcanoOutput[4]
    DRC_TS1_matrix = twoDvolcanoOutput[5]
    DRC_TS2_matrix = twoDvolcanoOutput[7]
    DRC_TS3_matrix = twoDvolcanoOutput[9]
    k1f_matrix = twoDvolcanoOutput[12]
    k2f_matrix = twoDvolcanoOutput[13]
    k3f_matrix = twoDvolcanoOutput[14]
    k_app_R_TS2_matrix = twoDvolcanoOutput[15]
    k_app_R_TS3_matrix = twoDvolcanoOutput[16]
    k_app_I1_TS3_matrix = twoDvolcanoOutput[17]
    maxSteadyStateRate_array = twoDvolcanoOutput[18]

    # Grab a sample extrapolation of rate constants
    k1f_sample_vary_BE = k1f_matrix[1,:]
    k2f_sample_vary_BE = k2f_matrix[1,:]
    k3f_sample_vary_BE = k3f_matrix[1,:]
    k_app_R_TS2_sample_vary_BE = k_app_R_TS2_matrix[1,:]
    k_app_R_TS3_sample_vary_BE = k_app_R_TS3_matrix[1,:]
    k_app_I1_TS3_sample_vary_BE = k_app_I1_TS3_matrix[1,:]
    k1f_sample_vary_E = k1f_matrix[:,1]
    k2f_sample_vary_E = k2f_matrix[:,1]
    k3f_sample_vary_E = k3f_matrix[:,1]
    k_app_R_TS2_sample_vary_E = k_app_R_TS2_matrix[:,1]
    k_app_R_TS3_sample_vary_E = k_app_R_TS3_matrix[:,1]
    k_app_I1_TS3_sample_vary_E = k_app_I1_TS3_matrix[:,1]

    X,Y = np.meshgrid(delta_H0_I1_array, E_array)

    # This will be the 3D surface with the extrapolated lines (i.e. rate constants) overlaid
    plt.figure(figsize=(12,12))
    ax = plt.axes(projection="3d",computed_zorder=false)
    ax.plot_surface(X,Y,log.(steadyStateRateMatrix), cmap="viridis",rstride=1,cstride=1,zorder=1)
    ax.set_xlabel(L"$\delta$H$^{0}_{I_{1}}$ (eV)", labelpad = 10, fontsize = 18)
    ax.set_ylabel("Potential (V)", labelpad = 10, fontsize = 18)
    ax.set_zlabel(L"$log(TOF)$", labelpad = 10, fontsize = 18)
    ax.tick_params(axis="x",labelsize=14,width=2)
    ax.tick_params(axis="y",labelsize=14,width=2)
    ax.tick_params(axis="z",labelsize=14,width=2)
    ax.set_yticks([0.1,0.2,0.3,0.4,0.5])
    ax.set_xlim([-0.75,0.25])
    ax.set_ylim([0.1,0.5])
    ax.grid(true)

    ax.plot(delta_H0_I1_array, E_array[1]*ones(length(E_array)), log.(k1f_sample_vary_BE),"r-", linewidth=4, zorder=3)
    ax.plot(delta_H0_I1_array, E_array[1]*ones(length(E_array)), log.(k_app_I1_TS3_sample_vary_BE),"r-", linewidth=4, zorder=3)
    # ax.plot(delta_H0_I1_array, E_array[1]*ones(length(E_array)), log.(k3f_sample_vary_BE),"r-", linewidth=2)  
    ax.plot(delta_H0_I1_array[1]*ones(length(delta_H0_I1_array)), E_array, log.(k3f_sample_vary_E),color=[0.0 0.0 0.4], linewidth=4, zorder=3)
    ax.plot(delta_H0_I1_array[1]*ones(length(delta_H0_I1_array)), E_array, log.(k_app_I1_TS3_sample_vary_E),color=[0.0 0.0 0.4], linewidth=4, zorder=3)
    # ax.plot(delta_H0_I1_array[1]*ones(length(delta_H0_I1_array)), E_array, log.(k3f_sample_vary_E),"b-", linewidth=2)  

    # To plot the line corresponding to the max rate on top of the 3D surface
    # Need to make an array of the BE values corresponding to the max TOF at each potential
    maxRate_BE_indices = zeros(length(E_array)) # initialize - will have one for every potential slice
    maxRate_vals = zeros(length(E_array)) # initialize - will have a max rate at each potential slice
    for i = 1:length(E_array)
        maxIndexArray = findall(steadyStateRateMatrix[i,:] .== maximum(steadyStateRateMatrix[i,:]))
        maxIndex = maxIndexArray[1]
        maxRate_BE_indices[i] = delta_H0_I1_array[maxIndex]
        maxRate_vals[i] = maximum(steadyStateRateMatrix[i,:])
    end
    ax.plot(maxRate_BE_indices,E_array,log.(maxRate_vals),color=[0.25,0.25,0.25],linewidth=2,zorder=2)

    # Plot the maximum rate points on the 3D TOF surface
    # for i = 1:length(E_array)
    #     maxIndexArray = findall(steadyStateRateMatrix[i,:] .== maximum(steadyStateRateMatrix[i,:]))
    #     maxIndex = maxIndexArray[1]
    #     # ax.scatter(delta_H0_I1_array[maxIndex],E_array[i],log.(maximum(steadyStateRateMatrix[i,:])),s = 10,c = "k", marker = ".", linewidths = 1,zorder=2)
    #     # ax.plot(X[maxIndex],Y[i],log.(maximum(steadyStateRateMatrix[i,:])),"k-", linewidth = 1,zorder=2)
    #     # ax.plot(delta_H0_I1_array[maxIndex]*ones(length(delta_H0_I1_array)),E_array,log10.(maximum(steadyStateRateMatrix[:,i])),"k-")
    # end


    # This will be the constant potential, variable binding energy slice corresponding to the extrapolated lines on the 3D surface

    fig, axs = plt.subplots(2,2,figsize=(13,9)) # need more spaces for plots when poisoning step is included
    ## Vary binding energy volcano ##
    axs[1,1].plot(delta_H0_I1_array, log.(steadyStateRateMatrix[1,:]),"k-",linewidth=3)
    axs[1,1].plot(delta_H0_I1_array, log.(k1f_sample_vary_BE), "bx", linewidth=3)
    axs[1,1].plot(delta_H0_I1_array, log.(k2f_sample_vary_BE), "b+", linewidth=3)
    axs[1,1].plot(delta_H0_I1_array, log.(k3f_sample_vary_BE), "b-", linewidth=3)
    #axs[1,1].plot(delta_H0_I1_array, log.(k2f_sample_vary_BE .* steadyStateI1CoverageMatrix[1,:]), "g--", linewidth=3)
    #axs[1,1].plot(delta_H0_I1_array, log.(k3f_sample_vary_BE .* steadyStateI2CoverageMatrix[1,:]), "gx", linewidth=3)
    #axs[1,1].plot(delta_H0_I1_array, log.(k_app_R_TS2_sample_vary_BE), "r-", linewidth=3)
    #axs[1,1].plot(delta_H0_I1_array, log.(k_app_R_TS3_sample_vary_BE), "r--", linewidth=3)
    axs[1,1].plot(delta_H0_I1_array, log.(k_app_I1_TS3_sample_vary_BE), "rx", linewidth=3)
    axs[1,1].set_ylabel(L"$log(TOF)$", fontsize=18)
    axs[1,1].tick_params(axis="x",labelsize=15, width=1.5)
    axs[1,1].tick_params(axis="y",labelsize=15, width=1.5)
    axs[1,1].legend(["log TOF","log k1f", "log k2f","log k3f", "log k_app_I1_TS3"])
    ## Vary potential volcano ##
    axs[1,2].plot(E_array, log.(steadyStateRateMatrix[:,1]),"k-",linewidth=3)
    axs[1,2].plot(E_array, log.(k1f_sample_vary_E), "bx", linewidth=3)
    axs[1,2].plot(E_array, log.(k2f_sample_vary_E), "b+", linewidth=3)
    axs[1,2].plot(E_array, log.(k3f_sample_vary_E), "b-", linewidth=3)
    #axs[1,2].plot(E_array, log.(k2f_sample_vary_E .* steadyStateI1CoverageMatrix[:,1]), "g--", linewidth=3)
    #axs[1,2].plot(E_array, log.(k3f_sample_vary_E .* steadyStateI2CoverageMatrix[:,1]), "gx", linewidth=3)
    axs[1,2].plot(E_array, log.(k_app_R_TS2_sample_vary_E), "r-", linewidth=3)
    axs[1,2].plot(E_array, log.(k_app_R_TS3_sample_vary_E), "r--", linewidth=3)
    axs[1,2].plot(E_array, log.(k_app_I1_TS3_sample_vary_E), "rx", linewidth=3)
    axs[1,2].legend(["log TOF","log k1f", "log k2f","log k3f", "log k_app_I1_TS3"])
    axs[1,2].tick_params(axis="x",labelsize=15, width=1.5)
    axs[1,2].tick_params(axis="y",labelsize=15, width=1.5)
    ## Coverages and DRCs for vary binding energy volcano ##
    axs[2,1].plot(delta_H0_I1_array, steadyStateI1CoverageMatrix[1,:],"k-",linewidth=3)
    axs[2,1].plot(delta_H0_I1_array, steadyStateI2CoverageMatrix[1,:],"kx",linewidth=3)
    axs[2,1].plot(delta_H0_I1_array, DRC_TS1_matrix[1,:],"c-",linewidth=3)
    axs[2,1].plot(delta_H0_I1_array, DRC_TS2_matrix[1,:],"c--",linewidth=3)
    axs[2,1].plot(delta_H0_I1_array, DRC_TS3_matrix[1,:],"cx",linewidth=3)
    axs[2,1].set_ylabel("DRC or Coverage", fontsize=18)
    axs[2,1].set_xlabel(L"$\delta$H$^{0}_{I_{1}}$ (eV)", fontsize = 18)
    axs[2,1].legend(["Cov I1","Cov I2","DRC TS1", "DRC TS2", "DRC TS3"])
    axs[2,1].tick_params(axis="x",labelsize=15, width=1.5)
    axs[2,1].tick_params(axis="y",labelsize=15, width=1.5)
    ## Coverages and DRCs for vary potential volcano ##
    axs[2,2].plot(E_array, steadyStateI1CoverageMatrix[:,1],"k-",linewidth=3)
    axs[2,2].plot(E_array, steadyStateI2CoverageMatrix[:,1],"kx",linewidth=3)
    axs[2,2].plot(E_array, DRC_TS1_matrix[:,1],"c-",linewidth=3)
    axs[2,2].plot(E_array, DRC_TS2_matrix[:,1],"c--",linewidth=3)
    axs[2,2].plot(E_array, DRC_TS3_matrix[:,1],"cx",linewidth=3)
    axs[2,2].set_xlabel("Potential (V)", fontsize = 18)
    axs[2,2].tick_params(axis="x",labelsize=15, width=1.5)
    axs[2,2].tick_params(axis="y",labelsize=15, width=1.5)
    #plt.ylabel("Coverage or DRC")
    axs[2,2].legend(["Cov I1","Cov I2","DRC TS1", "DRC TS2", "DRC TS3"])

    plt.show()


end

# make3Dfigure(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, np.linspace(-0.75,0.25,50), T, np.linspace(0.1,0.5,50), gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, [0.33,0.33,0.0], "Independent", "P")

# Try to make a "general" square wave function, which can oscillate either binding energy or potential depending on what the function input specifies
# The function will accept both binding energy (delta_H0_I1_array) and potential (E_array) as arrays
# For the quantity you wish to oscillate, just put two different values in the given array. For the quantity you wish to stay constant, just put the same value in both indices of the array
function squareWaveFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array, T, E_array, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionFirst, dutyCycle, frequency, initialConditionSteadyState)

    time2 = dutyCycle/(100*frequency) # time of the square wave "on" (weaker binding energy, or higher overpotential - i.e. state which favors desorption of accumulated intermediates)
    time1 = (1-(time2*frequency))/frequency # time of the square wave "off" (stronger binding energy, lower overpotential)

    # Solve the first ODE using the "first" initial condition (set in the outer function input) and the first binding energy/potential condition
    transientRateFunctionOutput_1 = transientRateFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[1], T, E_array[1], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, time1, initialConditionFirst)
    # transientRateFunctionOutput is [transientRate_P, transientRate_electron_tot, transientRate_electron_r1, transientRate_electron_r2, transientRate_electron_r3, t_I1, t_I2, t_Ix, t_star, timeArray]
    # Get the length of just the first leg of the oscillation (the lengths of those output variables will grow as things are appended onto them, so make a dummy variable here to get the length of just the first leg)
    transientRate_P_1_original = transientRateFunctionOutput_1[1]
    transientRate_electron_1_original = transientRateFunctionOutput_1[2]
    transientRate_electron_r1_1_original = transientRateFunctionOutput_1[3]
    transientRate_electron_r2_1_original = transientRateFunctionOutput_1[4]
    transientRate_electron_r3_1_original = transientRateFunctionOutput_1[5]
    transientRate_coverage_I1_1_original = transientRateFunctionOutput_1[6]
    transientRate_coverage_I2_1_original = transientRateFunctionOutput_1[7]
    transientRate_coverage_Ix_1_original = transientRateFunctionOutput_1[8]
    timeArray_1 = transientRateFunctionOutput_1[10] # actual time points from the ODE solution
    length_1 = length(transientRate_P_1_original)
    # Get the steady-state outputs as well - specify P or electron, but be able to grab both every time (don't need to specify in squareWaveFunction input)
    steadyStateRateFunctionOutput_P_1 = steadyStateRateFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[1], T, E_array[1], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionSteadyState, "P")
    steadyStateRate_P_1 = steadyStateRateFunctionOutput_P_1[1]
    steadyStateRateFunctionOutput_electron_1 = steadyStateRateFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[1], T, E_array[1], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionSteadyState, "electron")
    steadyStateRate_electron_1 = steadyStateRateFunctionOutput_electron_1[1]
    steadyState_coverage_I1_1 = steadyStateRateFunctionOutput_electron_1[2] # same for electron or P, just use electron
    steadyState_coverage_I2_1 = steadyStateRateFunctionOutput_electron_1[3] # same for electron or P, just use electron
    steadyState_coverage_Ix_1 = steadyStateRateFunctionOutput_electron_1[4] # same for electron or P, just use electron
    # steadyStateRateFunctionOutput is [steadyStateRate, steadyStateCoverageI1, steadyStateCoverageI2, steadyStateCoverageIx, stateStateCoverageStar] - note that these are "values" and not arrays - need to turn into an array with the length corresponding to the transient leg

    # Solve the second ODE using the "second" initial condition (which are the final coverages from the first integration) and the second binding energy/potential condition
    transientRateFunctionOutput_2 = transientRateFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[2], T, E_array[2], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, time2, [last(transientRate_coverage_I1_1_original), last(transientRate_coverage_I2_1_original), last(transientRate_coverage_Ix_1_original)])
    # transientRateFunctionOutput is [transientRate_P, transientRate_electron_tot, transientRate_electron_r1, transientRate_electron_r2, transientRate_electron_r3, t_I1, t_I2, t_Ix, t_star, timeArray]
    transientRate_P_2_original = transientRateFunctionOutput_2[1]
    transientRate_electron_2_original = transientRateFunctionOutput_2[2]
    transientRate_electron_r1_2_original = transientRateFunctionOutput_2[3]
    transientRate_electron_r2_2_original = transientRateFunctionOutput_2[4]
    transientRate_electron_r3_2_original = transientRateFunctionOutput_2[5]
    transientRate_coverage_I1_2_original = transientRateFunctionOutput_2[6]
    transientRate_coverage_I2_2_original = transientRateFunctionOutput_2[7]
    transientRate_coverage_Ix_2_original = transientRateFunctionOutput_2[8]
    timeArray_2 = transientRateFunctionOutput_2[10] # actual time points from the ODE solution
    length_2 = length(transientRate_P_2_original)
    # Get the steady-state outputs as well
    steadyStateRateFunctionOutput_P_2 = steadyStateRateFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[2], T, E_array[2], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionSteadyState, "P")
    steadyStateRate_P_2 = steadyStateRateFunctionOutput_P_2[1]
    steadyStateRateFunctionOutput_electron_2 = steadyStateRateFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[2], T, E_array[2], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionSteadyState, "electron")
    steadyStateRate_electron_2 = steadyStateRateFunctionOutput_electron_2[1]
    steadyState_coverage_I1_2 = steadyStateRateFunctionOutput_electron_2[2] # same for electron or P, just use electron
    steadyState_coverage_I2_2 = steadyStateRateFunctionOutput_electron_2[3] # same for electron or P, just use electron
    steadyState_coverage_Ix_2 = steadyStateRateFunctionOutput_electron_2[4] # same for electron or P, just use electron
    # steadyStateRateFunctionOutput is [steadyStateRate, steadyStateCoverageI1, steadyStateCoverageI2, steadyStateCoverageIx, stateStateCoverageStar] - note that these are "values" and not arrays - need to turn into an array with the length corresponding to the transient leg

    initialConditionNext = [last(transientRate_coverage_I1_2_original), last(transientRate_coverage_I2_2_original), last(transientRate_coverage_Ix_2_original)] # this will be a function output (initial condition for the next cycle when we start repeating oscillations)
    # Stitch together the outputs from the first and second legs of the oscillation
    # timeArrayTotal = append!(np.linspace(0.0, time1, length_1), np.linspace(time1, time1 + time2, length_2))

    # Instead, add timeArray_1 and timeArray_2 to get the total time per cycle
    timeArray_2_to_append = last(timeArray_1) .+ timeArray_2 # each ODE integration starts at "0" so need to add the second time Array onto the last time point of the prior time array
    timeArrayTotal = append!(timeArray_1, timeArray_2_to_append)

    dimensionlessTimeArrayTotal = timeArrayTotal./last(timeArrayTotal)
    # transientRateTotal = append!(transientRateFunctionOutput_1[1], transientRateFunctionOutput_2[1])
    # Try to concatenate instead of append to keep the original legs unchanged
    transientRate_P_Total = cat(transientRate_P_1_original, transientRate_P_2_original, dims=(1,1)) # this works better, keep it this way
    transientRate_electron_Total = cat(transientRate_electron_1_original, transientRate_electron_2_original, dims=(1,1)) # this works better, keep it this way
    transientRate_electron_r1_Total = cat(transientRate_electron_r1_1_original, transientRate_electron_r1_2_original, dims=(1,1)) # this works better, keep it this way
    transientRate_electron_r2_Total = cat(transientRate_electron_r2_1_original, transientRate_electron_r2_2_original, dims=(1,1)) # this works better, keep it this way
    transientRate_electron_r3_Total = cat(transientRate_electron_r3_1_original, transientRate_electron_r3_2_original, dims=(1,1)) # this works better, keep it this way
    coverageI1Total = append!(transientRate_coverage_I1_1_original, transientRate_coverage_I1_2_original)
    coverageI2Total = append!(transientRate_coverage_I2_1_original, transientRate_coverage_I2_2_original)
    coverageIxTotal = append!(transientRate_coverage_Ix_1_original, transientRate_coverage_Ix_2_original)
    steadyStateRate_P_Total = append!(steadyStateRate_P_1.*ones(length_1), steadyStateRate_P_2.*ones(length_2))
    steadyStateRate_electron_Total = append!(steadyStateRate_electron_1.*ones(length_1), steadyStateRate_electron_2.*ones(length_2))
    steadyStateCoverageI1Total = append!(steadyState_coverage_I1_1.*ones(length_1), steadyState_coverage_I1_2.*ones(length_2))
    steadyStateCoverageI2Total = append!(steadyState_coverage_I2_1.*ones(length_1), steadyState_coverage_I2_2.*ones(length_2))
    steadyStateCoverageIxTotal = append!(steadyState_coverage_Ix_1.*ones(length_1), steadyState_coverage_Ix_2.*ones(length_2))
    bindingEnergyTotal = append!(delta_H0_I1_array[1].*ones(length_1), delta_H0_I1_array[2].*ones(length_2))
    potentialTotal = append!(E_array[1].*ones(length_1), E_array[2].*ones(length_2))
    # Try to return each "leg" of the potential oscillation (the "on" and "off" or "low" and "high" legs)
    # This way we can calculate efficiencies WRT each leg of the oscillation to illustrate how efficiencies drop at higher potential
    potential_first_leg = E_array[1].*ones(length_1)
    potential_second_leg = E_array[2].*ones(length_2)
    # May also need the time arrays corresponding to each leg to do the efficiency calculation
    timeArray_first_leg = timeArray_1
    timeArray_second_leg = timeArray_2_to_append
    # And the rate terms as well...
    transientRate_P_first_leg = transientRate_P_1_original
    transientRate_P_second_leg = transientRate_P_2_original

    return initialConditionNext, timeArrayTotal, dimensionlessTimeArrayTotal, transientRate_P_Total, transientRate_electron_Total, transientRate_electron_r1_Total, transientRate_electron_r2_Total, transientRate_electron_r3_Total, coverageI1Total, coverageI2Total, coverageIxTotal, steadyStateRate_P_Total, steadyStateRate_electron_Total, steadyStateCoverageI1Total, steadyStateCoverageI2Total, steadyStateCoverageIxTotal, bindingEnergyTotal, potentialTotal, potential_first_leg, potential_second_leg, timeArray_first_leg, timeArray_second_leg, transientRate_P_first_leg, transientRate_P_second_leg
end

# This function will repeat the "squareWaveFunction" oscillations. It will first work through the "induction" period, which are the oscillations between the initial condition, and the stable dynamic steady-state condition. Sometimes it takes many cycles for the system to approach dynamic steady-state (i.e. when the integrated rate per cycle no longer changes, appreciably, with the number of cycles)
# We can also choose whether or not to make the plots independently in this function (plotMethod = "independent") or whether to just output/return the results to a different function to be worked with further or plotted there (plotMethod = "runFullAnalysis", or anything else other than "independent")
function repeatSquareWaveFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array, T, E_array, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionFirstOverall, dutyCycle, frequency, initialConditionSteadyState, numberOfDynamicSteadyStateOscillations, plotMethod, oscillationType) # need to specify oscillationType as "Potential" or "Binding Energy" for finding the "optimal" values for calculating enhancement relative to optimal steady-state

    # Begin the induction period by running the squareWaveFunction with the overall first initial condition
    inductionOutput = squareWaveFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array, T, E_array, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionFirstOverall, dutyCycle, frequency, initialConditionSteadyState)
    # "squareWaveFunction" output consists of [initialConditionNext, timeArrayTotal, dimensionlessTimeArrayTotal, transientRate_P_Total, transientRate_electron_Total, transientRate_electron_r1_Total, transientRate_electron_r2_Total, transientRate_electron_r3_Total, coverageI1Total, coverageI2Total, coverageIxTotal, steadyStateRate_P_Total, steadyStateRate_electron_Total, steadyStateCoverageI1Total, steadyStateCoverageI2Total, steadyStateCoverageIxTotal, bindingEnergyTotal, potentialTotal]
    # Still need to define these output variables before the while loop so that they can be properly updated when we append onto them
    initialConditionNext = inductionOutput[1]
    inductionTimeArray = inductionOutput[2]
    inductionDimensionlessTimeArray = inductionOutput[3]
    inductionTransientRate_P = inductionOutput[4]
    inductionTransientRate_electron = inductionOutput[5]
    inductionTransientRate_electron_r1 = inductionOutput[6]
    inductionTransientRate_electron_r2 = inductionOutput[7]
    inductionTransientRate_electron_r3 = inductionOutput[8]
    inductionCoverageI1 = inductionOutput[9]
    inductionCoverageI2 = inductionOutput[10]
    inductionCoverageIx = inductionOutput[11]
    inductionSteadyStateRate_P = inductionOutput[12]
    inductionSteadyStateRate_electron = inductionOutput[13]
    inductionSteadyStateCoverageI1 = inductionOutput[14]
    inductionSteadyStateCoverageI2 = inductionOutput[15]
    inductionSteadyStateCoverageIx = inductionOutput[16]
    inductionBindingEnergy = inductionOutput[17]
    inductionPotential = inductionOutput[18]
    inductionPotentialFirstLeg = inductionOutput[19]
    inductionPotentialSecondLeg = inductionOutput[20]
    inductionTimeArrayFirstLeg = inductionOutput[21]
    inductionTimeArraySecondLeg = inductionOutput[22]
    inductionTransientRate_P_first_leg = inductionOutput[23]
    inductionTransientRate_P_second_leg = inductionOutput[24]
    # Let's try to calculate some of the average and efficiency metrics during the induction period so we can plot them as a function of cycle # during the induction period. This can serve as a good sanity check that things are making sense.
    # Calculate the terms for the first induction period cycle and then append on the subsequent cycles inside the while loop, like we do for the other metrics
    inductionAverageDynamicRate_P = [convert(Float64,trapz(inductionTimeArray, inductionTransientRate_P) / (last(inductionTimeArray) - inductionTimeArray[1]))]
    inductionAverageDynamicRate_electron = [convert(Float64,trapz(inductionTimeArray, inductionTransientRate_electron) / (last(inductionTimeArray) - inductionTimeArray[1]))]
    inductionAverageSteadyStateRate_P = [convert(Float64,trapz(inductionTimeArray, inductionSteadyStateRate_P) / (last(inductionTimeArray) - inductionTimeArray[1]))]
    inductionAverageSteadyStateRate_electron = [convert(Float64,trapz(inductionTimeArray, inductionSteadyStateRate_electron) / (last(inductionTimeArray) - inductionTimeArray[1]))]
    # Also get an enhancement factor here
    inductionEnhancementFactorRelativeToSteadyState_P = [convert(Float64,(trapz(inductionTimeArray, inductionTransientRate_P) / (last(inductionTimeArray) - inductionTimeArray[1])) / (trapz(inductionTimeArray, inductionSteadyStateRate_P) / (last(inductionTimeArray) - inductionTimeArray[1])))]
    inductionEnhancementFactorRelativeToSteadyState_electron = [convert(Float64,(trapz(inductionTimeArray, inductionTransientRate_electron) / (last(inductionTimeArray) - inductionTimeArray[1])) / (trapz(inductionTimeArray, inductionSteadyStateRate_electron) / (last(inductionTimeArray) - inductionTimeArray[1])))]
    # For potential oscillations, also work out the electrochemical cell efficiency and its enhancement relative to steady-state during the induction period as well
    inductionElectrochemicalCellEfficiencyDynamic_P = [convert(Float64,(trapz(inductionTimeArray, (inductionTransientRate_P.*Eocv))/trapz(inductionTimeArray, (inductionTransientRate_P.*(Eocv .+ inductionPotential)))))]
    inductionElectrochemicalCellEfficiencyDynamic_electron = [convert(Float64,(trapz(inductionTimeArray, (inductionTransientRate_electron.*Eocv))/trapz(inductionTimeArray, (inductionTransientRate_electron.*(Eocv .+ inductionPotential)))))]
    inductionElectrochemicalCellEfficiencySteadyState_P = [convert(Float64,(trapz(inductionTimeArray, (inductionSteadyStateRate_P.*Eocv))/trapz(inductionTimeArray, (inductionSteadyStateRate_P.*(Eocv .+ inductionPotential)))))]
    inductionElectrochemicalCellEfficiencySteadyState_electron = [convert(Float64,(trapz(inductionTimeArray, (inductionSteadyStateRate_electron.*Eocv))/trapz(inductionTimeArray, (inductionSteadyStateRate_electron.*(Eocv .+ inductionPotential)))))]
    # Calculate the efficiencies during the induction period corresponding to each leg of the potential oscillation
    # println(length(inductionTimeArrayFirstLeg))
    # println(length(inductionTransientRate_P_first_leg))
    # println(length(inductionPotentialFirstLeg))
    # println(length(inductionTimeArraySecondLeg))
    # println(length(inductionTransientRate_P_second_leg))
    # println(length(inductionPotentialSecondLeg))
    # inductionElectrochemicalCellEfficiencyDynamicFirstLeg_P = [convert(Float64,(trapz(inductionTimeArrayFirstLeg, (inductionTransientRate_P_first_leg.*Eocv))/trapz(inductionTimeArrayFirstLeg, (inductionTransientRate_P_first_leg.*(Eocv .+ inductionPotentialFirstLeg)))))]
    # inductionElectrochemicalCellEfficiencyDynamicSecondLeg_P = [convert(Float64,(trapz(inductionTimeArraySecondLeg, (inductionTransientRate_P_second_leg.*Eocv))/trapz(inductionTimeArraySecondLeg, (inductionTransientRate_P_second_leg.*(Eocv .+ inductionPotentialSecondLeg)))))]

    # Need to make a WHILE loop for the induction period
    # The induction period is comprised of the oscillations between the overall initial condition and the point where the system reaches a "dynamic steady-state"
    # While the integrated rate of a given cycle is not equal to the integrated rate of the previous cycle, keep going
    # Once they are equal (within a certain percentage tolerance), then exit the while loop, and begin the "actual" dynamic catalysis analysis
    integratedTransientRate_P_CurrentCycle = trapz(inductionTimeArray, inductionTransientRate_P)
    integratedTransientRate_P_PreviousCycle = integratedTransientRate_P_CurrentCycle + 10 # This is a dummy placeholder, force this to be greater than the first cycle so that the while loop always initiates
    integratedTransientRate_electron_CurrentCycle = trapz(inductionTimeArray, inductionTransientRate_electron)
    integratedTransientRate_electron_PreviousCycle = integratedTransientRate_electron_CurrentCycle + 10 # This is a dummy placeholder, force this to be greater than the first cycle so that the while loop always initiates
    inductionCycleCount = 0 # count the number of cycles needed to complete the induction period
    inductionTolerance = 1e-13 # integrated charge of the current cycle must be within 1E-11% of the integrated charge of the previous cycle
    # inductionTolerance = 1e-6 # test for higher freq. ranges

    while abs((integratedTransientRate_P_CurrentCycle - integratedTransientRate_P_PreviousCycle)/integratedTransientRate_P_PreviousCycle) > inductionTolerance && abs((integratedTransientRate_electron_CurrentCycle - integratedTransientRate_electron_PreviousCycle)/integratedTransientRate_electron_PreviousCycle) > inductionTolerance
        # println(inductionCycleCount) # print the cycle number to keep track of how fast we are moving through the induction period
        integratedTransientRate_P_PreviousCycle = integratedTransientRate_P_CurrentCycle # after a cycle, the "current" cycle of cycle "i" becomes the "previous" of cycle "i+1"
        integratedTransientRate_electron_PreviousCycle = integratedTransientRate_electron_CurrentCycle # after a cycle, the "current" cycle of cycle "i" becomes the "previous" of cycle "i+1"
        # Run a single oscillation cycle, using the "initialConditionNext"
        singleOscillationOutput = squareWaveFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array, T, E_array, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionNext, dutyCycle, frequency, initialConditionSteadyState)
        # "squareWaveFunction" output consists of [initialConditionNext, timeArrayTotal, dimensionlessTimeArrayTotal, transientRate_P_Total, transientRate_electron_Total, transientRate_electron_r1_Total, transientRate_electron_r2_Total, transientRate_electron_r3_Total, coverageI1Total, coverageI2Total, coverageIxTotal, steadyStateRate_P_Total, steadyStateRate_electron_Total, steadyStateCoverageI1Total, steadyStateCoverageI2Total, steadyStateCoverageIxTotal, bindingEnergyTotal, potentialTotal]

        # Need to add the time arrays (real and dimensionless) as the cycles progress
        realTimeToAddPerOscillation = last(inductionTimeArray) .+ singleOscillationOutput[2]
        dimensionlessTimeToAddPerOscillation = last(inductionDimensionlessTimeArray) .+ singleOscillationOutput[3]

        # Get the integrated rate of the current cycle. These are calculated here for the purpose of setting the condition for the while loop.
        # Don't need to do for single electron rates since they all should approach DSS at the same time, since coverages are identical
        integratedTransientRate_P_CurrentCycle = trapz(singleOscillationOutput[2], singleOscillationOutput[4]) # integrated with respect to real time, for product P
        integratedTransientRate_electron_CurrentCycle = trapz(singleOscillationOutput[2], singleOscillationOutput[5]) # integrated with respect to real time, for electrons

        # Add the current cycle rates/coverages/times onto the running array of previous cycles to stitch together the entire induction period
        append!(inductionTimeArray, realTimeToAddPerOscillation)
        append!(inductionDimensionlessTimeArray, dimensionlessTimeToAddPerOscillation)
        append!(inductionTransientRate_P, singleOscillationOutput[4])
        append!(inductionTransientRate_electron, singleOscillationOutput[5])
        append!(inductionTransientRate_electron_r1, singleOscillationOutput[6])
        append!(inductionTransientRate_electron_r2, singleOscillationOutput[7])
        append!(inductionTransientRate_electron_r3, singleOscillationOutput[8])
        append!(inductionCoverageI1, singleOscillationOutput[9])
        append!(inductionCoverageI2, singleOscillationOutput[10])
        append!(inductionCoverageIx, singleOscillationOutput[11])
        append!(inductionSteadyStateRate_P, singleOscillationOutput[12])
        append!(inductionSteadyStateRate_electron, singleOscillationOutput[13])
        append!(inductionSteadyStateCoverageI1, singleOscillationOutput[14])
        append!(inductionSteadyStateCoverageI2, singleOscillationOutput[15])
        append!(inductionSteadyStateCoverageIx, singleOscillationOutput[16])

        # Work out the "average" and "enhancement" terms inside here
        # These terms will just be a number per each cycle, as opposed to an array per each cycle like the other metrics, since they are evaluated by averaging the data over each cycle
        # Because of that, we need to plot them vs. the counter # and not versus the time (or dimensionless time) variables like the other terms
        # Have this function output the total counter variable for a given oscillation condition, then create a linspaced array up to that counter for plotting these metrics later
        append!(inductionAverageDynamicRate_P, trapz(realTimeToAddPerOscillation, singleOscillationOutput[4]) / (last(realTimeToAddPerOscillation) - realTimeToAddPerOscillation[1]))
        append!(inductionAverageDynamicRate_electron, trapz(realTimeToAddPerOscillation, singleOscillationOutput[5]) / (last(realTimeToAddPerOscillation) - realTimeToAddPerOscillation[1]))
        append!(inductionAverageSteadyStateRate_P, trapz(realTimeToAddPerOscillation, singleOscillationOutput[12]) / (last(realTimeToAddPerOscillation) - realTimeToAddPerOscillation[1]))
        append!(inductionAverageSteadyStateRate_electron, trapz(realTimeToAddPerOscillation, singleOscillationOutput[13]) / (last(realTimeToAddPerOscillation) - realTimeToAddPerOscillation[1]))
        append!(inductionEnhancementFactorRelativeToSteadyState_P, ((trapz(realTimeToAddPerOscillation, singleOscillationOutput[4]) / (last(realTimeToAddPerOscillation) - realTimeToAddPerOscillation[1]))/(trapz(realTimeToAddPerOscillation, singleOscillationOutput[12]) / (last(realTimeToAddPerOscillation) - realTimeToAddPerOscillation[1]))))
        append!(inductionEnhancementFactorRelativeToSteadyState_electron, ((trapz(realTimeToAddPerOscillation, singleOscillationOutput[5]) / (last(realTimeToAddPerOscillation) - realTimeToAddPerOscillation[1]))/(trapz(realTimeToAddPerOscillation, singleOscillationOutput[13]) / (last(realTimeToAddPerOscillation) - realTimeToAddPerOscillation[1]))))
        # Also update the electrochemical efficiency terms during the induction period
        append!(inductionElectrochemicalCellEfficiencyDynamic_P, trapz(realTimeToAddPerOscillation, (singleOscillationOutput[4].*Eocv))/trapz(realTimeToAddPerOscillation, (singleOscillationOutput[4].*(Eocv .+ singleOscillationOutput[18]))))
        append!(inductionElectrochemicalCellEfficiencyDynamic_electron, trapz(realTimeToAddPerOscillation, (singleOscillationOutput[5].*Eocv))/trapz(realTimeToAddPerOscillation, (singleOscillationOutput[5].*(Eocv .+ singleOscillationOutput[18]))))
        append!(inductionElectrochemicalCellEfficiencySteadyState_P, trapz(realTimeToAddPerOscillation, (singleOscillationOutput[12].*Eocv))/trapz(realTimeToAddPerOscillation, (singleOscillationOutput[12].*(Eocv .+ singleOscillationOutput[18]))))
        append!(inductionElectrochemicalCellEfficiencySteadyState_P, trapz(realTimeToAddPerOscillation, (singleOscillationOutput[13].*Eocv))/trapz(realTimeToAddPerOscillation, (singleOscillationOutput[13].*(Eocv .+ singleOscillationOutput[18]))))
        # Separate the efficiencies into the first and second legs of the potential oscillation
        # append!(inductionElectrochemicalCellEfficiencyDynamicFirstLeg_P, trapz(singleOscillationOutput[21], (singleOscillationOutput[23].*Eocv))/trapz(singleOscillationOutput[21], (singleOscillationOutput[23].*(Eocv .+ singleOscillationOutput[19]))))
        # append!(inductionElectrochemicalCellEfficiencyDynamicSecondLeg_P, trapz(singleOscillationOutput[22], (singleOscillationOutput[24].*Eocv))/trapz(singleOscillationOutput[22], (singleOscillationOutput[24].*(Eocv .+ singleOscillationOutput[20]))))

        # Update the initial condition and the cycle counter
        initialConditionNext = singleOscillationOutput[1]
        inductionCycleCount += 1
    end

    # Now start to perform the dynamic catalysis (i.e. at the dynamic steady state)
    dynamicSteadyStateOutput = squareWaveFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array, T, E_array, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionNext, dutyCycle, frequency, initialConditionSteadyState)
    # "squareWaveFunction" output consists of [initialConditionNext, timeArrayTotal, dimensionlessTimeArrayTotal, transientRate_P_Total, transientRate_electron_Total, transientRate_electron_r1_Total, transientRate_electron_r2_Total, transientRate_electron_r3_Total, coverageI1Total, coverageI2Total, coverageIxTotal, steadyStateRate_P_Total, steadyStateRate_electron_Total, steadyStateCoverageI1Total, steadyStateCoverageI2Total, steadyStateCoverageIxTotal, bindingEnergyTotal, potentialTotal]
    # Still need to define these output variables before the while loop so that they can be properly updated when we append onto them
    initialConditionNext = dynamicSteadyStateOutput[1]
    dynamicSteadyStateTimeArray = dynamicSteadyStateOutput[2]

    dynamicSteadyStateTimeArrayFirstCycle = dynamicSteadyStateTimeArray
    lengthFirstDynamicSteadyStateCycle = length(dynamicSteadyStateTimeArrayFirstCycle) # we will use this so that no matter how many cycles are "repeated" for plotting the dynamic Steady State rates, we can always grab just a single (i.e. the first) cycle. We want to use this for plotting the dynamic TOFs overlaid on the steady-state TOF volcanos at both binding energies/potentials

    dynamicSteadyStateDimensionlessTimeArray = dynamicSteadyStateOutput[3]
    dynamicSteadyStateTransientRate_P = dynamicSteadyStateOutput[4]
    dynamicSteadyStateTransientRate_electron = dynamicSteadyStateOutput[5]
    dynamicSteadyStateTransientRate_electron_r1 = dynamicSteadyStateOutput[6]
    dynamicSteadyStateTransientRate_electron_r2 = dynamicSteadyStateOutput[7]
    dynamicSteadyStateTransientRate_electron_r3 = dynamicSteadyStateOutput[8]
    dynamicSteadyStateCoverageI1 = dynamicSteadyStateOutput[9]
    dynamicSteadyStateCoverageI2 = dynamicSteadyStateOutput[10]
    dynamicSteadyStateCoverageIx = dynamicSteadyStateOutput[11]
    # Also get the outputs for a single oscillation under oscillatory steady-state conditions
    oscillatorySteadyStateRate_P = dynamicSteadyStateOutput[12]
    oscillatorySteadyStateRate_electron = dynamicSteadyStateOutput[13]
    oscillatorySteadyStateCoverageI1 = dynamicSteadyStateOutput[14]
    oscillatorySteadyStateCoverageI2 = dynamicSteadyStateOutput[15]
    oscillatorySteadyStateCoverageIx = dynamicSteadyStateOutput[16]
    # Get the binding energies and potentials for plotting their oscillations
    bindingEnergyOscillation = dynamicSteadyStateOutput[17]
    potentialOscillation = dynamicSteadyStateOutput[18]

    # Calculate the average coverage per cycle to use as initial guesses when running simulations at different frequencies
    averageCoverageI1 = trapz(dynamicSteadyStateTimeArray, dynamicSteadyStateCoverageI1) / (last(dynamicSteadyStateTimeArray) - dynamicSteadyStateTimeArray[1])
    averageCoverageI2 = trapz(dynamicSteadyStateTimeArray, dynamicSteadyStateCoverageI2) / (last(dynamicSteadyStateTimeArray) - dynamicSteadyStateTimeArray[1])
    averageCoverageIx = trapz(dynamicSteadyStateTimeArray, dynamicSteadyStateCoverageIx) / (last(dynamicSteadyStateTimeArray) - dynamicSteadyStateTimeArray[1])

    # Can get the "average" terms based on a single cycle at dynamic steady state (since the cycles are technically "identical" within a certain tolerance from here on out)
    averageDynamicSteadyStateTransientRate_P = trapz(dynamicSteadyStateTimeArray, dynamicSteadyStateTransientRate_P)/(last(dynamicSteadyStateTimeArray) - dynamicSteadyStateTimeArray[1])
    averageDynamicSteadyStateTransientRate_electron = trapz(dynamicSteadyStateTimeArray, dynamicSteadyStateTransientRate_electron)/(last(dynamicSteadyStateTimeArray) - dynamicSteadyStateTimeArray[1])
    # Get the average oscillatory steady-state terms as well
    averageOscillatorySteadyStateRate_P = trapz(dynamicSteadyStateTimeArray, oscillatorySteadyStateRate_P)/(last(dynamicSteadyStateTimeArray) - dynamicSteadyStateTimeArray[1])
    averageOscillatorySteadyStateRate_electron = trapz(dynamicSteadyStateTimeArray, oscillatorySteadyStateRate_electron)/(last(dynamicSteadyStateTimeArray) - dynamicSteadyStateTimeArray[1])

    # Get enhancement factors relative to oscillatory steady state
    enhancementFactorRelativeToOscillatorySteadyState_P = averageDynamicSteadyStateTransientRate_P/averageOscillatorySteadyStateRate_P
    enhancementFactorRelativeToOscillatorySteadyState_electron = averageDynamicSteadyStateTransientRate_electron/averageOscillatorySteadyStateRate_electron

    # Get enhancement factors relative to optimal steady state. Need to get optimal steady states for P and electrons, and for either potential or binding energy oscillation types, from different functions, over the potential/binding energy range being oscillated between
    # steadyStateRateFunctionVary... output is [steadyStateRateArray, steadyStateI1CoverageArray, steadyStateI2CoverageArray, steadyStateIxCoverageArray, maxSteadyStateRate, DRC_TS1_array, DRC_I1_array, DRC_TS2_array, DRC_I2_array, DRC_TS3_array, DRC_TSx_array, DRC_Ix_array]
    if oscillationType == "Potential"
        steadyStateRateFunctionVaryPotentialOutput_P = steadyStateRateFunctionVaryPotential(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[1], T, np.linspace(E_array[1], E_array[2], 50), gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionSteadyState, plotMethod, "P") # keep binding energy constant  
        optimalSteadyStateRate_P = steadyStateRateFunctionVaryPotentialOutput_P[5]
        steadyStateRateFunctionVaryPotentialOutput_electron = steadyStateRateFunctionVaryPotential(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[1], T, np.linspace(E_array[1], E_array[2], 50), gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionSteadyState, plotMethod, "electron") # keep binding energy constant
        optimalSteadyStateRate_electron = steadyStateRateFunctionVaryPotentialOutput_electron[5]
    elseif oscillationType == "Binding Energy"
        steadyStateRateFunctionVaryBindingEnergyOutput_P = steadyStateRateFunctionVaryBindingEnergy(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, np.linspace(delta_H0_I1_array[1], delta_H0_I1_array[2], 50), T, E_array[1], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionSteadyState, plotMethod, "P") # keep potential constant  
        optimalSteadyStateRate_P = steadyStateRateFunctionVaryBindingEnergyOutput_P[5]
        steadyStateRateFunctionVaryBindingEnergyOutput_electron = steadyStateRateFunctionVaryBindingEnergy(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, np.linspace(delta_H0_I1_array[1], delta_H0_I1_array[2], 50), T, E_array[1], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionSteadyState, plotMethod, "electron") # keep potential constant  
        optimalSteadyStateRate_electron = steadyStateRateFunctionVaryBindingEnergyOutput_electron[5]
    end
    enhancementFactorRelativeToOptimalSteadyState_P = averageDynamicSteadyStateTransientRate_P/optimalSteadyStateRate_P
    enhancementFactorRelativeToOptimalSteadyState_electron = averageDynamicSteadyStateTransientRate_electron/optimalSteadyStateRate_electron
    # Try to get the Dauenhauer "turnover efficiency" here
    turnoverEfficiency_P = (averageDynamicSteadyStateTransientRate_P - averageOscillatorySteadyStateRate_P)/frequency
    turnoverEfficiency_electron = (averageDynamicSteadyStateTransientRate_electron - averageOscillatorySteadyStateRate_electron)/frequency
    # Try to get the electrochemical efficiencies here
    electrochemicalCellEfficiencyDynamic_P = trapz(dynamicSteadyStateTimeArray, (dynamicSteadyStateTransientRate_P.*Eocv))/trapz(dynamicSteadyStateTimeArray, (dynamicSteadyStateTransientRate_P.*(Eocv .+ potentialOscillation)))
    electrochemicalCellEfficiencyDynamic_electron = trapz(dynamicSteadyStateTimeArray, (dynamicSteadyStateTransientRate_electron.*Eocv))/trapz(dynamicSteadyStateTimeArray, (dynamicSteadyStateTransientRate_electron.*(Eocv .+ potentialOscillation)))
    electrochemicalCellEfficiencyOscillatorySteadyState_P = trapz(dynamicSteadyStateTimeArray, (oscillatorySteadyStateRate_P.*Eocv))/trapz(dynamicSteadyStateTimeArray, (oscillatorySteadyStateRate_P.*(Eocv .+ potentialOscillation)))
    electrochemicalCellEfficiencyOscillatorySteadyState_electron = trapz(dynamicSteadyStateTimeArray, (oscillatorySteadyStateRate_electron.*Eocv))/trapz(dynamicSteadyStateTimeArray, (oscillatorySteadyStateRate_electron.*(Eocv .+ potentialOscillation)))

    # Get the electrochemical cell efficiency enhancement factors here
    electrochemicalCellEfficiencyEnhancementFactor_P = electrochemicalCellEfficiencyDynamic_P - electrochemicalCellEfficiencyOscillatorySteadyState_P
    electrochemicalCellEfficiencyEnhancementFactor_electron = electrochemicalCellEfficiencyDynamic_electron - electrochemicalCellEfficiencyOscillatorySteadyState_electron

    # Below will be just for plotting. Want to repeat a few cycles at dynamic steady-state to see how it looks. Can also plot the oscillatory steady-state as well on top of it, with coverages and binding energies too.
    for i in 1:numberOfDynamicSteadyStateOscillations
        repeatDynamicSteadyStateOutput = squareWaveFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array, T, E_array, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionNext, dutyCycle, frequency, initialConditionSteadyState)
        # "squareWaveFunction" output consists of [initialConditionNext, timeArrayTotal, dimensionlessTimeArrayTotal, transientRate_P_Total, transientRate_electron_Total, transientRate_electron_r1_Total, transientRate_electron_r2_Total, transientRate_electron_r3_Total, coverageI1Total, coverageI2Total, coverageIxTotal, steadyStateRate_P_Total, steadyStateRate_electron_Total, steadyStateCoverageI1Total, steadyStateCoverageI2Total, steadyStateCoverageIxTotal, bindingEnergyTotal, potentialTotal]
        # Add the time arrays as the cycles repeat
        repeatDynamicSteadyStateTimeToAddPerOscillation = last(dynamicSteadyStateTimeArray) .+ repeatDynamicSteadyStateOutput[2]
        repeatDynamicSteadyStateDimensionlessTimeToAddPerOscillation = last(dynamicSteadyStateDimensionlessTimeArray) .+ repeatDynamicSteadyStateOutput[3]
        # Add current cycle rates/coverages/times
        append!(dynamicSteadyStateTimeArray, repeatDynamicSteadyStateTimeToAddPerOscillation)
        append!(dynamicSteadyStateDimensionlessTimeArray, repeatDynamicSteadyStateDimensionlessTimeToAddPerOscillation)
        append!(dynamicSteadyStateTransientRate_P, repeatDynamicSteadyStateOutput[4])
        append!(dynamicSteadyStateTransientRate_electron, repeatDynamicSteadyStateOutput[5])
        append!(dynamicSteadyStateTransientRate_electron_r1, repeatDynamicSteadyStateOutput[6])
        append!(dynamicSteadyStateTransientRate_electron_r2, repeatDynamicSteadyStateOutput[7])
        append!(dynamicSteadyStateTransientRate_electron_r3, repeatDynamicSteadyStateOutput[8])
        append!(dynamicSteadyStateCoverageI1, repeatDynamicSteadyStateOutput[9])
        append!(dynamicSteadyStateCoverageI2, repeatDynamicSteadyStateOutput[10])
        append!(dynamicSteadyStateCoverageIx, repeatDynamicSteadyStateOutput[11])
        append!(oscillatorySteadyStateRate_P, repeatDynamicSteadyStateOutput[12])
        append!(oscillatorySteadyStateRate_electron, repeatDynamicSteadyStateOutput[13])
        append!(oscillatorySteadyStateCoverageI1, repeatDynamicSteadyStateOutput[14])
        append!(oscillatorySteadyStateCoverageI2, repeatDynamicSteadyStateOutput[15])
        append!(oscillatorySteadyStateCoverageIx, repeatDynamicSteadyStateOutput[16])
        append!(bindingEnergyOscillation, repeatDynamicSteadyStateOutput[17])
        append!(potentialOscillation, repeatDynamicSteadyStateOutput[18])

        # Update the initial condition
        initialConditionNext = repeatDynamicSteadyStateOutput[1]
    end

    ## UNCOMMENT THIS SECTION IF YOU WANT TO PLOT THE INDUCTION PERIOD EVERYTIME THIS FUNCTION IS CALLED ##
    # Plot the finished induction period everytime this is called
    # plt.figure(1)
    # plt.plot(inductionTimeArray, inductionCoverageI1,"k-", inductionTimeArray, inductionCoverageI2,"b-", inductionTimeArray,inductionCoverageIx,"r-",inductionTimeArray, inductionSteadyStateCoverageI1,"k:", inductionTimeArray, inductionSteadyStateCoverageI2,"b:", inductionTimeArray,inductionSteadyStateCoverageIx,"r:", linewidth = 3)
    # plt.xlabel("TIme (s)", fontsize = 18)
    # plt.ylabel("Induction Period Coverage", labelpad = 8.0, fontsize = 18)
    # plt.yticks([0,0.5,1.0])
    # plt.tick_params(axis="x",labelsize=16, width=1.5)
    # plt.tick_params(axis="y",labelsize=16, width=1.5)
    # # plt.title("A",loc="left",fontsize=18,fontweight="bold")
    # plt.legend(["I1", "I2", "Ix"], loc = "upper right", fontsize = 18)
    # plt.show()


    ## UNCOMMENT THIS SECTION IF YOU WANT TO KEEP OVERLAYING INDUCTION PERIODS (FOR EXAMPLE WHEN CALLING THE varyFrequencyConstantAmplitude FUNCTION)
    # plt.plot(inductionTimeArray, inductionTransientRate_P,"-", inductionTimeArray, inductionSteadyStateRate_P,":", linewidth = 3)
    # plt.xlabel("Time (s)", fontsize = 18)
    # plt.ylabel("TOF", labelpad = 8.0, fontsize = 18)
    # # plt.yticks([0,0.5,1.0])
    # plt.tick_params(axis="x",labelsize=16, width=1.5)
    # plt.tick_params(axis="y",labelsize=16, width=1.5)
    # # plt.title("A",loc="left",fontsize=18,fontweight="bold")
    # plt.legend(["Dyn P", "SS_P",], loc = "upper right", fontsize = 18)

    return dynamicSteadyStateTimeArray, dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateTransientRate_P, dynamicSteadyStateTransientRate_electron, dynamicSteadyStateTransientRate_electron_r1, dynamicSteadyStateTransientRate_electron_r2, dynamicSteadyStateTransientRate_electron_r3, dynamicSteadyStateCoverageI1, dynamicSteadyStateCoverageI2, dynamicSteadyStateCoverageIx, oscillatorySteadyStateRate_P, oscillatorySteadyStateRate_electron, oscillatorySteadyStateCoverageI1, oscillatorySteadyStateCoverageI2, oscillatorySteadyStateCoverageIx, bindingEnergyOscillation, potentialOscillation, initialConditionNext, inductionTimeArray, inductionDimensionlessTimeArray, inductionTransientRate_P, inductionTransientRate_electron, inductionTransientRate_electron_r1, inductionTransientRate_electron_r2, inductionTransientRate_electron_r3, inductionCoverageI1, inductionCoverageI2, inductionCoverageIx, inductionSteadyStateRate_P, inductionSteadyStateRate_electron, inductionSteadyStateCoverageI1, inductionSteadyStateCoverageI2, inductionSteadyStateCoverageIx, averageDynamicSteadyStateTransientRate_P, averageDynamicSteadyStateTransientRate_electron, averageOscillatorySteadyStateRate_P, averageOscillatorySteadyStateRate_electron, enhancementFactorRelativeToOscillatorySteadyState_P, enhancementFactorRelativeToOscillatorySteadyState_electron, enhancementFactorRelativeToOptimalSteadyState_P, enhancementFactorRelativeToOptimalSteadyState_electron, turnoverEfficiency_P, turnoverEfficiency_electron, electrochemicalCellEfficiencyDynamic_P, electrochemicalCellEfficiencyDynamic_electron, electrochemicalCellEfficiencyOscillatorySteadyState_P, electrochemicalCellEfficiencyOscillatorySteadyState_electron, electrochemicalCellEfficiencyEnhancementFactor_P, electrochemicalCellEfficiencyEnhancementFactor_electron, inductionCycleCount, inductionAverageDynamicRate_P, inductionAverageDynamicRate_electron, inductionAverageSteadyStateRate_P, inductionAverageSteadyStateRate_electron, inductionEnhancementFactorRelativeToSteadyState_P, inductionEnhancementFactorRelativeToSteadyState_electron, inductionElectrochemicalCellEfficiencyDynamic_P, inductionElectrochemicalCellEfficiencyDynamic_electron, inductionElectrochemicalCellEfficiencySteadyState_P, inductionElectrochemicalCellEfficiencySteadyState_electron, averageCoverageI1, averageCoverageI2, averageCoverageIx
end

# This function will run "repeatSquareWaveFunction" over a series of potential/binding energy amplitudes and frequencies in order to generate heat maps of the desired output variables from "repeatSquareWaveFunction," i.e. enhancement factors, efficiencies, etc
# If you want to keep a variable constant (binding energy or potential) just set its amplitude array to all 0's, while the variable you oscillate will have a non-zero amplitude. Still need to indicate "oscillationType" as "Potential" or "Binding Energy" as well, for plotting and grabbing the "optimal" steady-state rates from the "steadyState..." functions
function makeOscillationHeatMap(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_center, delta_H0_I1_amplitude_array, T, E_lower, E_amplitude_array, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionFirstOverall, dutyCycle, frequencyArray, initialConditionSteadyState, numberOfDynamicSteadyStateOscillations, oscillationType, plotMethod)

    # Initialize matrices to store various enhancement/efficiency values over frequency and binding energy/potential values
    enhancementFactorRelativeToOscillatorySteadyState_P_matrix = zeros(length(delta_H0_I1_amplitude_array), length(frequencyArray))
    enhancementFactorRelativeToOscillatorySteadyState_electron_matrix = zeros(length(delta_H0_I1_amplitude_array), length(frequencyArray))
    enhancementFactorRelativeToOptimalSteadyState_P_matrix = zeros(length(delta_H0_I1_amplitude_array), length(frequencyArray))
    enhancementFactorRelativeToOptimalSteadyState_electron_matrix = zeros(length(delta_H0_I1_amplitude_array), length(frequencyArray))
    turnoverEfficiency_P_matrix = zeros(length(delta_H0_I1_amplitude_array), length(frequencyArray))
    turnoverEfficiency_electron_matrix = zeros(length(delta_H0_I1_amplitude_array), length(frequencyArray))
    electrochemicalCellEfficiencyDynamic_P_matrix = zeros(length(delta_H0_I1_amplitude_array), length(frequencyArray))
    electrochemicalCellEfficiencyDynamic_electron_matrix = zeros(length(delta_H0_I1_amplitude_array), length(frequencyArray))
    electrochemicalCellEfficiencyOscillatorySteadyState_P_matrix = zeros(length(delta_H0_I1_amplitude_array), length(frequencyArray))
    electrochemicalCellEfficiencyOscillatorySteadyState_electron_matrix = zeros(length(delta_H0_I1_amplitude_array), length(frequencyArray)) 
    electrochemicalEfficiencyEnhancementFactor_P_matrix = zeros(length(delta_H0_I1_amplitude_array), length(frequencyArray))
    electrochemicalEfficiencyEnhancementFactor_electron_matrix = zeros(length(delta_H0_I1_amplitude_array), length(frequencyArray))
    initialCondition = initialConditionFirstOverall

    for i in 1:length(delta_H0_I1_amplitude_array) # again, same length as E_amplitude_array, so doesn't matter whether we're oscillating binding energy or potential
        for j in 1:length(frequencyArray) 
            #tick()
            repeatSquareWaveFunctionOutput = repeatSquareWaveFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, [delta_H0_I1_center - (delta_H0_I1_amplitude_array[i]), delta_H0_I1_center + (delta_H0_I1_amplitude_array[i])], T, [E_lower, E_lower + (E_amplitude_array[i])], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, dutyCycle, frequencyArray[j], initialConditionSteadyState, numberOfDynamicSteadyStateOscillations, plotMethod, oscillationType)
            #tock()
            # repeatSquareWaveFunction output is [dynamicSteadyStateTimeArray, dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateTransientRate_P, dynamicSteadyStateTransientRate_electron, dynamicSteadyStateTransientRate_electron_r1, dynamicSteadyStateTransientRate_electron_r2, dynamicSteadyStateTransientRate_electron_r3, dynamicSteadyStateCoverageI1, dynamicSteadyStateCoverageI2, dynamicSteadyStateCoverageIx, oscillatorySteadyStateRate_P, oscillatorySteadyStateRate_electron, oscillatorySteadyStateCoverageI1, oscillatorySteadyStateCoverageI2, oscillatorySteadyStateCoverageIx, bindingEnergyOscillation, potentialOscillation, initialConditionNext, inductionTimeArray, inductionDimensionlessTimeArray, inductionTransientRate_P, inductionTransientRate_electron, inductionTransientRate_electron_r1, inductionTransientRate_electron_r2, inductionTransientRate_electron_r3, inductionCoverageI1, inductionCoverageI2, inductionCoverageIx, inductionSteadyStateRate_P, inductionSteadyStateRate_electron, inductionSteadyStateCoverageI1, inductionSteadyStateCoverageI2, inductionSteadyStateCoverageIx, averageDynamicSteadyStateTransientRate_P, averageDynamicSteadyStateTransientRate_electron, averageOscillatorySteadyStateRate_P, averageOscillatorySteadyStateRate_electron, enhancementFactorRelativeToOscillatorySteadyState_P, enhancementFactorRelativeToOscillatorySteadyState_electron, enhancementFactorRelativeToOptimalSteadyState_P, enhancementFactorRelativeToOptimalSteadyState_electron, turnoverEfficiency_P, turnoverEfficiency_electron, electrochemicalCellEfficiencyDynamic_P, electrochemicalCellEfficiencyDynamic_electron, electrochemicalCellEfficiencyOscillatorySteadyState_P, electrochemicalCellEfficiencyOscillatorySteadyState_electron, electrochemicalCellEfficiencyEnhancementFactor_P, electrochemicalCellEfficiencyEnhancementFactor_electron, inductionCycleCount, inductionAverageDynamicRate_P, inductionAverageDynamicRate_electron, inductionAverageSteadyStateRate_P, inductionAverageSteadyStateRate_electron, inductionEnhancementFactorRelativeToSteadyState_P, inductionEnhancementFactorRelativeToSteadyState_electron, inductionElectrochemicalCellEfficiencyDynamic_P, inductionElectrochemicalCellEfficiencyDynamic_electron, inductionElectrochemicalCellEfficiencySteadyState_P, inductionElectrochemicalCellEfficiencySteadyState_electron, averageCoverageI1, averageCoverageI2, averageCoverageIx]
            enhancementFactorRelativeToOscillatorySteadyState_P_matrix[j,i] = repeatSquareWaveFunctionOutput[38]
            enhancementFactorRelativeToOscillatorySteadyState_electron_matrix[j,i] = repeatSquareWaveFunctionOutput[39]
            enhancementFactorRelativeToOptimalSteadyState_P_matrix[j,i] = repeatSquareWaveFunctionOutput[40]
            enhancementFactorRelativeToOptimalSteadyState_electron_matrix[j,i] = repeatSquareWaveFunctionOutput[41]
            turnoverEfficiency_P_matrix[j,i] = repeatSquareWaveFunctionOutput[42]
            turnoverEfficiency_electron_matrix[j,i] = repeatSquareWaveFunctionOutput[43]
            electrochemicalCellEfficiencyDynamic_P_matrix[j,i] = repeatSquareWaveFunctionOutput[44]
            electrochemicalCellEfficiencyDynamic_electron_matrix[j,i] = repeatSquareWaveFunctionOutput[45]
            electrochemicalCellEfficiencyOscillatorySteadyState_P_matrix[j,i] = repeatSquareWaveFunctionOutput[46]
            electrochemicalCellEfficiencyOscillatorySteadyState_electron_matrix[j,i] = repeatSquareWaveFunctionOutput[47]
            electrochemicalEfficiencyEnhancementFactor_P_matrix[j,i] = repeatSquareWaveFunctionOutput[48]
            electrochemicalEfficiencyEnhancementFactor_electron_matrix[j,i] = repeatSquareWaveFunctionOutput[49]
        end
    end

    return enhancementFactorRelativeToOscillatorySteadyState_P_matrix, enhancementFactorRelativeToOscillatorySteadyState_electron_matrix, enhancementFactorRelativeToOptimalSteadyState_P_matrix, enhancementFactorRelativeToOptimalSteadyState_electron_matrix, turnoverEfficiency_P_matrix, turnoverEfficiency_electron_matrix, electrochemicalEfficiencyEnhancementFactor_P_matrix, electrochemicalEfficiencyEnhancementFactor_electron_matrix
end

# Make a function to generate plots of the avg TOF, or the enhancement factors, as a function of frequency but at a single oscillation amplitude
function varyFrequencyConstantAmplitude(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_center, delta_H0_I1_amplitude, T, E_lower, E_amplitude, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionFirstOverall, dutyCycle, frequencyArray, initialConditionSteadyState, numberOfDynamicSteadyStateOscillations, oscillationType, plotMethod)

    enhancementFactorRelativeToOscillatorySteadyState_P_array = zeros(length(frequencyArray))
    enhancementFactorRelativeToOptimalSteadyState_P_array = zeros(length(frequencyArray))
    initialCondition = initialConditionFirstOverall
    
    for i = 1:length(frequencyArray)
        if i == 1
            repeatSquareWaveFunctionOutput = repeatSquareWaveFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, [delta_H0_I1_center - (delta_H0_I1_amplitude), delta_H0_I1_center + (delta_H0_I1_amplitude)], T, [E_lower, E_lower + (E_amplitude)], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, dutyCycle, frequencyArray[i], initialConditionSteadyState, numberOfDynamicSteadyStateOscillations, plotMethod, oscillationType)
            enhancementFactorRelativeToOscillatorySteadyState_P_array[i] = repeatSquareWaveFunctionOutput[38]
            enhancementFactorRelativeToOptimalSteadyState_P_array[i] = repeatSquareWaveFunctionOutput[40]
            # initialCondition = repeatSquareWaveFunctionOutput[18]
            averageCoverageI1 = repeatSquareWaveFunctionOutput[61]
            averageCoverageI2 = repeatSquareWaveFunctionOutput[62]
            averageCoverageIx = repeatSquareWaveFunctionOutput[63]
            initialCondition = [averageCoverageI1, averageCoverageI2, averageCoverageIx]

        else

            repeatSquareWaveFunctionOutput = repeatSquareWaveFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, [delta_H0_I1_center - (delta_H0_I1_amplitude), delta_H0_I1_center + (delta_H0_I1_amplitude)], T, [E_lower, E_lower + (E_amplitude)], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialCondition, dutyCycle, frequencyArray[i], initialConditionSteadyState, numberOfDynamicSteadyStateOscillations, plotMethod, oscillationType)
            enhancementFactorRelativeToOscillatorySteadyState_P_array[i] = repeatSquareWaveFunctionOutput[38]
            enhancementFactorRelativeToOptimalSteadyState_P_array[i] = repeatSquareWaveFunctionOutput[40]
            # initialCondition = repeatSquareWaveFunctionOutput[18]
            averageCoverageI1 = repeatSquareWaveFunctionOutput[61]
            averageCoverageI2 = repeatSquareWaveFunctionOutput[62]
            averageCoverageIx = repeatSquareWaveFunctionOutput[63]
            initialCondition = [averageCoverageI1, averageCoverageI2, averageCoverageIx]

        end
    end

    return enhancementFactorRelativeToOscillatorySteadyState_P_array, enhancementFactorRelativeToOptimalSteadyState_P_array
end

# Make a full analysis function. Makes steady-state volcanos, and dynamic catalysis outputs. Can also make heatMaps if you indicate in the function input (which will then use amplitude/frequency arrays accordingly)
function runFullAnalysis(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array, T, E_array, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionFirstOverall, dutyCycle, frequency, initialConditionSteadyState, numberOfDynamicSteadyStateOscillations, oscillationType, productType, volcanosOn, dynamicCatalysisOn, heatMapsOn, frequencyArray, delta_H0_I1_center, delta_H0_I1_amplitude_array, E_lower, E_amplitude_array, poisoningSpecies, plotInductionPeriod) # 3 main subplots can be made - make them by inputting "yes" for volcanosOn, dynamicCatalysisOn, and heatMapsOn

    if volcanosOn == "yes"
        ##### MAKE THE FIRST SUBPLOT: STEADY-STATE TOF, COVERAGE, AND DRC VOLCANOS #####
        ################################################################################
        fig, axs = plt.subplots(2,4,figsize=(22,14)) # need more spaces for plots when poisoning step is included
        plt.subplots_adjust(hspace = -0.55, wspace = 0.15, left = 0.06, right = 0.94, bottom = 0, top = 1.0)
        delta_H0_I1_array_for_SS_volcanos = np.linspace(-0.5,0.5,50) # range for 2D volcano plots
        E_array_for_SS_volcanos = np.linspace(0,0.75,50) # range for 2D volcano plots
        twoDimensionalVolcanoOutputs = makeTwoDimensionalPotentialAndBindingEnergyVolcano(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array_for_SS_volcanos, T, E_array_for_SS_volcanos, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionSteadyState, "runFullAnalysis", productType)
        # makeTwoDimensionalPotentialAndBindingEnergyVolcano outputs are: [steadyStateRateMatrix, steadyStateI1CoverageMatrix, steadyStateI2CoverageMatrix, steadyStateIxCoverageMatrix, DRC_TS1_matrix, DRC_I1_matrix, DRC_TS2_matrix, DRC_I2_matrix, DRC_TS3_matrix, DRC_TSx_matrix, DRC_Ix_matrix]
        steadyStateRateMatrix = twoDimensionalVolcanoOutputs[1]
        steadyStateCoverageI1Matrix = twoDimensionalVolcanoOutputs[2]
        steadyStateCoverageI2Matrix = twoDimensionalVolcanoOutputs[3]
        steadyStateCoverageIxMatrix = twoDimensionalVolcanoOutputs[4]
        DRC_TS1_matrix = twoDimensionalVolcanoOutputs[5]
        DRC_I1_matrix = twoDimensionalVolcanoOutputs[6]
        DRC_TS2_matrix = twoDimensionalVolcanoOutputs[7]
        DRC_I2_matrix = twoDimensionalVolcanoOutputs[8]
        DRC_TS3_matrix = twoDimensionalVolcanoOutputs[9]
        DRC_TSx_matrix = twoDimensionalVolcanoOutputs[10]
        DRC_Ix_matrix = twoDimensionalVolcanoOutputs[11]
        # Indicate the markers showing the bounds between which you are oscillating
        oscillationBindingEnergyBounds = delta_H0_I1_array
        oscillationPotentialBounds = E_array
        oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot = [0.0,0.0]
        oscillationPotentialBoundsForPotentialOscillationForThesisPlot = [0.1,0.5]
        X,Y = np.meshgrid(delta_H0_I1_array_for_SS_volcanos, E_array_for_SS_volcanos)
        # Subplot #1: steady-state log TOF volcano
        f1 = axs[1,1].contourf(X,Y,log.(steadyStateRateMatrix),levels=np.linspace(-7.5,maximum(log.(steadyStateRateMatrix)),50),extend="both",cmap="viridis")
        axs[1,1].set_ylabel("Potential (V)", fontsize = 20, fontname = "Arial")
        axs[1,1].set_xticks([])
        axs[1,1].set_yticks([0,0.2,0.4,0.6,0.8])
        axs[1,1].tick_params(axis="x",labelsize=15, width=1.5)
        axs[1,1].tick_params(axis="y",labelsize=15, width=1.5)
        axs[1,1].set_title("A",loc="left",fontsize=20,fontweight="bold")
        cb1 = plt.colorbar(f1,ax=axs[1,1], shrink = 0.30)
        cb1.ax.set_title(label=L"$log(TOF)$", fontsize = 20, pad = 18)
        cb1.set_ticks([-5,0,5])
        cb1.ax.tick_params(labelsize=14)
        axs[1,1].scatter(oscillationBindingEnergyBounds[1], oscillationPotentialBounds[1], s = 250, c = "k", marker = "x", linewidths = 5)
        axs[1,1].scatter(oscillationBindingEnergyBounds[2], oscillationPotentialBounds[2], s = 250, c = "k", marker = "x", linewidths = 5)
        # For making the extra SI figure with a bunch of potential amplitudes overlaid on same plot
        # axs[1,1].scatter(-0.25, 0.1, s = 100, c = "w", marker = "x", linewidths = 5)
        # axs[1,1].scatter(-0.25, 0.3, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[1,1].scatter(-0.25, 0.4, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[1,1].scatter(-0.25, 0.5, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[1,1].scatter(-0.25, 0.6, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[1,1].scatter(oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot[1], oscillationPotentialBoundsForPotentialOscillationForThesisPlot[1], s = 250, c = "k", marker = "x", linewidths = 5)
        # axs[1,1].scatter(oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot[2], oscillationPotentialBoundsForPotentialOscillationForThesisPlot[2], s = 250, c = "k", marker = "x", linewidths = 5)
        axs[1,1].axes.set_aspect("equal")
        # Subplot #2: steady-state coverage I1 / -DRC I1 volcano
        f2 = axs[1,2].contourf(X,Y,steadyStateCoverageI1Matrix,levels=np.linspace(0,1,50),extend="both",cmap="viridis")
        axs[1,2].set_xticks([])
        axs[1,2].set_yticks([])
        axs[1,2].set_xticklabels([])
        axs[1,2].set_yticklabels([])
        axs[1,2].set_title("B",loc="left",fontsize=20,fontweight="bold")
        cb2 = plt.colorbar(f2,ax=axs[1,2], shrink = 0.30)
        cb2.ax.set_title(label=L"$\theta_{I_{1}}(-X_{I_{1}})$", pad = 18, fontsize = 20, fontname = "monospace", fontstyle = "italic", fontweight = "bold")
        cb2.set_ticks([0,0.5,1.0])
        cb2.ax.tick_params(labelsize=14)
        axs[1,2].scatter(oscillationBindingEnergyBounds[1], oscillationPotentialBounds[1], s = 250, c = "k", marker = "x", linewidths = 5)
        axs[1,2].scatter(oscillationBindingEnergyBounds[2], oscillationPotentialBounds[2], s = 250, c = "k", marker = "x", linewidths = 5)
        # For making the extra SI figure with a bunch of potential amplitudes overlaid on same plot
        # axs[1,2].scatter(-0.25, 0.1, s = 100, c = "w", marker = "x", linewidths = 5)
        # axs[1,2].scatter(-0.25, 0.3, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[1,2].scatter(-0.25, 0.4, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[1,2].scatter(-0.25, 0.5, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[1,2].scatter(-0.25, 0.6, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[1,2].scatter(oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot[1], oscillationPotentialBoundsForPotentialOscillationForThesisPlot[1], s = 250, c = "k", marker = "x", linewidths = 5)
        # axs[1,2].scatter(oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot[2], oscillationPotentialBoundsForPotentialOscillationForThesisPlot[2], s = 250, c = "k", marker = "x", linewidths = 5)
        axs[1,2].axes.set_aspect("equal")
        # Subplot #3: steady-state coverage I2 / -DRC I2 volcano
        f3 = axs[1,3].contourf(X,Y,steadyStateCoverageI2Matrix,levels=np.linspace(0,1,50),extend="both",cmap="viridis")
        axs[1,3].set_xticks([])
        axs[1,3].set_yticks([])
        axs[1,3].set_xticklabels([])
        axs[1,3].set_yticklabels([])
        axs[1,3].set_title("C",loc="left",fontsize=20,fontweight="bold")
        cb3 = plt.colorbar(f3,ax=axs[1,3], shrink = 0.30)
        cb3.ax.set_title(label=L"$\theta_{I_{2}}(-X_{I_{2}})$",  pad = 18, fontsize = 20, fontname = "monospace", fontstyle = "italic", fontweight = "bold")
        cb3.set_ticks([0,0.5,1.0])
        cb3.ax.tick_params(labelsize=14)
        axs[1,3].scatter(oscillationBindingEnergyBounds[1], oscillationPotentialBounds[1], s = 250, c = "k", marker = "x", linewidths = 5)
        axs[1,3].scatter(oscillationBindingEnergyBounds[2], oscillationPotentialBounds[2], s = 250, c = "k", marker = "x", linewidths = 5)
        # For making the extra SI figure with a bunch of potential amplitudes overlaid on same plot
        # axs[1,3].scatter(-0.25, 0.1, s = 100, c = "w", marker = "x", linewidths = 5)
        # axs[1,3].scatter(-0.25, 0.3, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[1,3].scatter(-0.25, 0.4, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[1,3].scatter(-0.25, 0.5, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[1,3].scatter(-0.25, 0.6, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[1,3].scatter(oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot[1], oscillationPotentialBoundsForPotentialOscillationForThesisPlot[1], s = 250, c = "k", marker = "x", linewidths = 5)
        # axs[1,3].scatter(oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot[2], oscillationPotentialBoundsForPotentialOscillationForThesisPlot[2], s = 250, c = "k", marker = "x", linewidths = 5)
        axs[1,3].axes.set_aspect("equal")
        # Subplot #4: steady-state coverage Ix / -DRC Ix volcano
        f4 = axs[1,4].contourf(X,Y,steadyStateCoverageIxMatrix,levels=np.linspace(0,1,50),extend="both",cmap="viridis")
        axs[1,4].set_xticks([])
        axs[1,4].set_yticks([])
        axs[1,4].set_xticklabels([])
        axs[1,4].set_yticklabels([])
        axs[1,4].set_title("D",loc="left",fontsize=20,fontweight="bold")
        cb4 = plt.colorbar(f4,ax=axs[1,4], shrink = 0.30)
        cb4.ax.set_title(label=L"$\theta_{I_{X}}(-X_{I_{X}})$",  pad = 18, fontsize = 20, fontname = "monospace", fontstyle = "italic", fontweight = "bold")
        cb4.set_ticks([0,0.5,1.0])
        cb4.ax.tick_params(labelsize=14)
        axs[1,4].scatter(oscillationBindingEnergyBounds[1], oscillationPotentialBounds[1], s = 250, c = "k", marker = "x", linewidths = 5)
        axs[1,4].scatter(oscillationBindingEnergyBounds[2], oscillationPotentialBounds[2], s = 250, c = "k", marker = "x", linewidths = 5)
        # For making the extra SI figure with a bunch of potential amplitudes overlaid on same plot
        # axs[1,4].scatter(-0.25, 0.1, s = 100, c = "w", marker = "x", linewidths = 5)
        # axs[1,4].scatter(-0.25, 0.3, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[1,4].scatter(-0.25, 0.4, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[1,4].scatter(-0.25, 0.5, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[1,4].scatter(-0.25, 0.6, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[1,4].scatter(oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot[1], oscillationPotentialBoundsForPotentialOscillationForThesisPlot[1], s = 250, c = "k", marker = "x", linewidths = 5)
        # axs[1,4].scatter(oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot[2], oscillationPotentialBoundsForPotentialOscillationForThesisPlot[2], s = 250, c = "k", marker = "x", linewidths = 5)
        axs[1,4].axes.set_aspect("equal")
        # Subplot #5: steady-state DRC TS1
        f5 = axs[2,1].contourf(X,Y,DRC_TS1_matrix,levels=np.linspace(0,1,50),extend="both",cmap="viridis")
        axs[2,1].set_xlabel(L"$\delta$H$^{0}_{I_{1}}$ (eV)", labelpad = 8.0, fontsize = 20)
        axs[2,1].set_ylabel("Potential (V)", fontsize = 20)
        axs[2,1].set_xticks([-0.5,0,0.5])
        axs[2,1].set_yticks([0,0.2,0.4,0.6,0.8])
        axs[2,1].tick_params(axis="x",labelsize=15, width=1.5)
        axs[2,1].tick_params(axis="y",labelsize=15, width=1.5)
        axs[2,1].set_title("D",loc="left",fontsize=20,fontweight="bold")
        cb5 = plt.colorbar(f5,ax=axs[2,1], shrink = 0.30)
        cb5.ax.set_title(label=L"$X_{TS_{1}}$",  pad = 18, fontsize = 20)
        cb5.set_ticks([0,0.5,1.0])
        cb5.ax.tick_params(labelsize=14)
        axs[2,1].scatter(oscillationBindingEnergyBounds[1], oscillationPotentialBounds[1], s = 250, c = "k", marker = "x", linewidths = 5)
        axs[2,1].scatter(oscillationBindingEnergyBounds[2], oscillationPotentialBounds[2], s = 250, c = "k", marker = "x", linewidths = 5)
        # For making the extra SI figure with a bunch of potential amplitudes overlaid on same plot
        # axs[2,1].scatter(-0.25, 0.1, s = 100, c = "w", marker = "x", linewidths = 5)
        # axs[2,1].scatter(-0.25, 0.3, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[2,1].scatter(-0.25, 0.4, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[2,1].scatter(-0.25, 0.5, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[2,1].scatter(-0.25, 0.6, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[2,1].scatter(oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot[1], oscillationPotentialBoundsForPotentialOscillationForThesisPlot[1], s = 250, c = "k", marker = "x", linewidths = 5)
        # axs[2,1].scatter(oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot[2], oscillationPotentialBoundsForPotentialOscillationForThesisPlot[2], s = 250, c = "k", marker = "x", linewidths = 5)
        axs[2,1].axes.set_aspect("equal")
        # Subplot #6: steady-state DRC TS2
        f6 = axs[2,2].contourf(X,Y,DRC_TS2_matrix,levels=np.linspace(0,1,50),extend="both",cmap="viridis")
        axs[2,2].set_xlabel(L"$\delta$H$^{0}_{I_{1}}$ (eV)", labelpad = 8.0, fontsize = 20)
        axs[2,2].set_xticks([-0.5,0,0.5])
        axs[2,2].set_yticks([])
        axs[2,2].tick_params(axis="x",labelsize=15, width=1.5)
        axs[2,2].tick_params(axis="y",labelsize=15, width=1.5)
        axs[2,2].set_title("E",loc="left",fontsize=20,fontweight="bold")
        cb6 = plt.colorbar(f6,ax=axs[2,2], shrink = 0.30)
        cb6.ax.set_title(label=L"$X_{TS_{2}}$",  pad = 18, fontsize = 20)
        cb6.set_ticks([0,0.5,1.0])
        cb6.ax.tick_params(labelsize=14)
        axs[2,2].scatter(oscillationBindingEnergyBounds[1], oscillationPotentialBounds[1], s = 250, c = "k", marker = "x", linewidths = 5)
        axs[2,2].scatter(oscillationBindingEnergyBounds[2], oscillationPotentialBounds[2], s = 250, c = "k", marker = "x", linewidths = 5)
        # For making the extra SI figure with a bunch of potential amplitudes overlaid on same plot
        # axs[2,2].scatter(-0.25, 0.1, s = 100, c = "w", marker = "x", linewidths = 5)
        # axs[2,2].scatter(-0.25, 0.3, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[2,2].scatter(-0.25, 0.4, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[2,2].scatter(-0.25, 0.5, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[2,2].scatter(-0.25, 0.6, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[2,2].scatter(oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot[1], oscillationPotentialBoundsForPotentialOscillationForThesisPlot[1], s = 250, c = "k", marker = "x", linewidths = 5)
        # axs[2,2].scatter(oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot[2], oscillationPotentialBoundsForPotentialOscillationForThesisPlot[2], s = 250, c = "k", marker = "x", linewidths = 5)
        axs[2,2].axes.set_aspect("equal")
        # Subplot #7: steady-state DRC TS3
        f7 = axs[2,3].contourf(X,Y,DRC_TS3_matrix,levels=np.linspace(0,1,50),extend="both",cmap="viridis")
        axs[2,3].set_xlabel(L"$\delta$H$^{0}_{I_{1}}$ (eV)", labelpad = 8.0, fontsize = 20)
        axs[2,3].set_xticks([-0.5,0,0.5])
        axs[2,3].set_yticks([])
        axs[2,3].tick_params(axis="x",labelsize=15, width=1.5)
        axs[2,3].tick_params(axis="y",labelsize=15, width=1.5)
        axs[2,3].set_title("F",loc="left",fontsize=20,fontweight="bold")
        cb7 = plt.colorbar(f7,ax=axs[2,3], shrink = 0.30)
        cb7.ax.set_title(label=L"$X_{TS_{3}}$",  pad = 18, fontsize = 20)
        cb7.set_ticks([0,0.5,1.0])
        cb7.ax.tick_params(labelsize=14)
        axs[2,3].scatter(oscillationBindingEnergyBounds[1], oscillationPotentialBounds[1], s = 250, c = "k", marker = "x", linewidths = 5)
        axs[2,3].scatter(oscillationBindingEnergyBounds[2], oscillationPotentialBounds[2], s = 250, c = "k", marker = "x", linewidths = 5)
        # For making the extra SI figure with a bunch of potential amplitudes overlaid on same plot
        # axs[2,3].scatter(-0.25, 0.1, s = 100, c = "w", marker = "x", linewidths = 5)
        # axs[2,3].scatter(-0.25, 0.3, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[2,3].scatter(-0.25, 0.4, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[2,3].scatter(-0.25, 0.5, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[2,3].scatter(-0.25, 0.6, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[2,3].scatter(oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot[1], oscillationPotentialBoundsForPotentialOscillationForThesisPlot[1], s = 250, c = "k", marker = "x", linewidths = 5)
        # axs[2,3].scatter(oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot[2], oscillationPotentialBoundsForPotentialOscillationForThesisPlot[2], s = 250, c = "k", marker = "x", linewidths = 5)
        axs[2,3].axes.set_aspect("equal")
        # Subplot #8: steady-state DRC TSx
        f8 = axs[2,4].contourf(X,Y,DRC_TSx_matrix,levels=np.linspace(0,1,50),extend="both",cmap="viridis")
        axs[2,4].set_xlabel(L"$\delta$H$^{0}_{I_{1}}$ (eV)", labelpad = 8.0, fontsize = 20)
        axs[2,4].set_xticks([-0.5,0,0.5])
        axs[2,4].set_yticks([])
        axs[2,4].tick_params(axis="x",labelsize=15, width=1.5)
        axs[2,4].tick_params(axis="y",labelsize=15, width=1.5)
        axs[2,4].set_title("H",loc="left",fontsize=20,fontweight="bold")
        cb8 = plt.colorbar(f8,ax=axs[2,4], shrink = 0.30)
        cb8.ax.set_title(label=L"$X_{TS_{X}}$",  pad = 20, fontsize = 20)
        cb8.set_ticks([0,0.5,1.0])
        cb8.ax.tick_params(labelsize=14)
        axs[2,4].scatter(oscillationBindingEnergyBounds[1], oscillationPotentialBounds[1], s = 250, c = "k", marker = "x", linewidths = 5)
        axs[2,4].scatter(oscillationBindingEnergyBounds[2], oscillationPotentialBounds[2], s = 250, c = "k", marker = "x", linewidths = 5)
        # For making the extra SI figure with a bunch of potential amplitudes overlaid on same plot
        # axs[2,4].scatter(-0.25, 0.1, s = 100, c = "w", marker = "x", linewidths = 5)
        # axs[2,4].scatter(-0.25, 0.3, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[2,4].scatter(-0.25, 0.4, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[2,4].scatter(-0.25, 0.5, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[2,4].scatter(-0.25, 0.6, s = 100, c = "k", marker = "x", linewidths = 5)
        # axs[2,4].scatter(oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot[1], oscillationPotentialBoundsForPotentialOscillationForThesisPlot[1], s = 250, c = "k", marker = "x", linewidths = 5)
        # axs[2,4].scatter(oscillationBindingEnergyBoundsForPotentialOscillationForThesisPlot[2], oscillationPotentialBoundsForPotentialOscillationForThesisPlot[2], s = 250, c = "k", marker = "x", linewidths = 5)
        axs[2,4].axes.set_aspect("equal")
        plt.show()
    end

    if dynamicCatalysisOn == "yes"
        ##### MAKE THE SECOND SUBPLOT: SINGLE OSCILLATION CONDITION DYNAMIC CATALYSIS OUTPUTS #####
        ################################################################################
        fig, axs = plt.subplots(2,2,figsize=(9,7))
        fig.tight_layout(pad=6.0)
        # Subplot #1: Free energy diagrams for the two oscillation conditions. The plotting details for this subplot will be inside the "makeEnergyDiagram" function, so check in there
        if oscillationType == "Potential"
            color_array = ["b","k"] # have potential inputs always increasing, so that blue is lower E (adsorption), black is higher E (desorption)
            for i in 1:length(E_array)
                makeEnergyDiagram(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[1], T, E_array[i], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, "Gibbs", color_array[i], "runFullAnalysis", axs) # delta_H0_I1_array[1] = delta_H0_I1_array[2] if oscillating potential (constant binding energy)
            end
        elseif oscillationType == "Binding Energy"
            color_array = ["b","k"] # have binding energy inputs be increasing, so that strong binding (i.e. adsorption) is blue, weak binding (i.e. desorption) is black
            for i in 1:length(delta_H0_I1_array)
                makeEnergyDiagram(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array[i], T, E_array[1], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, "Gibbs", color_array[i], "runFullAnalysis", axs) # E_array[1] = E_array[2] if oscillating binding energy (constant potential)
            end
        end
        # Need to get the repeatSquareWaveFunction outputs now
        repeatSquareWaveFunctionOutput = repeatSquareWaveFunction(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_array, T, E_array, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionFirstOverall, dutyCycle, frequency, initialConditionSteadyState, numberOfDynamicSteadyStateOscillations, "runFullAnalysis", oscillationType)
        # repeatSquareWaveFunction output is [dynamicSteadyStateTimeArray, dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateTransientRate_P, dynamicSteadyStateTransientRate_electron, dynamicSteadyStateTransientRate_electron_r1, dynamicSteadyStateTransientRate_electron_r2, dynamicSteadyStateTransientRate_electron_r3, dynamicSteadyStateCoverageI1, dynamicSteadyStateCoverageI2, dynamicSteadyStateCoverageIx, oscillatorySteadyStateRate_P, oscillatorySteadyStateRate_electron, oscillatorySteadyStateCoverageI1, oscillatorySteadyStateCoverageI2, oscillatorySteadyStateCoverageIx, bindingEnergyOscillation, potentialOscillation, initialConditionNext, inductionTimeArray, inductionDimensionlessTimeArray, inductionTransientRate_P, inductionTransientRate_electron, inductionTransientRate_electron_r1, inductionTransientRate_electron_r2, inductionTransientRate_electron_r3, inductionCoverageI1, inductionCoverageI2, inductionCoverageIx, inductionSteadyStateRate_P, inductionSteadyStateRate_electron, inductionSteadyStateCoverageI1, inductionSteadyStateCoverageI2, inductionSteadyStateCoverageIx, averageDynamicSteadyStateTransientRate_P, averageDynamicSteadyStateTransientRate_electron, averageOscillatorySteadyStateRate_P, averageOscillatorySteadyStateRate_electron, enhancementFactorRelativeToOscillatorySteadyState_P, enhancementFactorRelativeToOscillatorySteadyState_electron, enhancementFactorRelativeToOptimalSteadyState_P, enhancementFactorRelativeToOptimalSteadyState_electron, turnoverEfficiency_P, turnoverEfficiency_electron, electrochemicalCellEfficiencyDynamic_P, electrochemicalCellEfficiencyDynamic_electron, electrochemicalCellEfficiencyOscillatorySteadyState_P, electrochemicalCellEfficiencyOscillatorySteadyState_electron, electrochemicalCellEfficiencyEnhancementFactor_P, electrochemicalCellEfficiencyEnhancementFactor_electron, inductionCycleCount, inductionAverageDynamicRate_P, inductionAverageDynamicRate_electron, inductionAverageSteadyStateRate_P, inductionAverageSteadyStateRate_electron, inductionEnhancementFactorRelativeToSteadyState_P, inductionEnhancementFactorRelativeToSteadyState_electron, inductionElectrochemicalCellEfficiencyDynamic_P, inductionElectrochemicalCellEfficiencyDynamic_electron, inductionElectrochemicalCellEfficiencySteadyState_P, inductionElectrochemicalCellEfficiencySteadyState_electron, averageCoverageI1, averageCoverageI2, averageCoverageIx]
        dynamicSteadyStateTimeArray = repeatSquareWaveFunctionOutput[1]
        dynamicSteadyStateDimensionlessTimeArray = repeatSquareWaveFunctionOutput[2]
        dynamicSteadyStateTransientRate_P = repeatSquareWaveFunctionOutput[3]
        dynamicSteadyStateTransientRate_electron = repeatSquareWaveFunctionOutput[4]
        dynamicSteadyStateTransientRate_electron_r1 = repeatSquareWaveFunctionOutput[5]
        dynamicSteadyStateTransientRate_electron_r2 = repeatSquareWaveFunctionOutput[6]
        dynamicSteadyStateTransientRate_electron_r3 = repeatSquareWaveFunctionOutput[7]
        dynamicSteadyStateCoverageI1 = repeatSquareWaveFunctionOutput[8]
        dynamicSteadyStateCoverageI2 = repeatSquareWaveFunctionOutput[9]
        dynamicSteadyStateCoverageIx = repeatSquareWaveFunctionOutput[10]
        oscillatorySteadyStateRate_P = repeatSquareWaveFunctionOutput[11]
        oscillatorySteadyStateRate_electron = repeatSquareWaveFunctionOutput[12]
        oscillatorySteadyStateCoverageI1 = repeatSquareWaveFunctionOutput[13]
        oscillatorySteadyStateCoverageI2 = repeatSquareWaveFunctionOutput[14]
        oscillatorySteadyStateCoverageIx = repeatSquareWaveFunctionOutput[15]
        bindingEnergyOscillation = repeatSquareWaveFunctionOutput[16]
        potentialOscillation = repeatSquareWaveFunctionOutput[17]
        inductionTimeArray = repeatSquareWaveFunctionOutput[19] 
        inductionDimensionlessTimeArray = repeatSquareWaveFunctionOutput[20]
        inductionTransientRate_P = repeatSquareWaveFunctionOutput[21]
        inductionTransientRate_electron = repeatSquareWaveFunctionOutput[22]
        inductionTransientRate_electron_r1 = repeatSquareWaveFunctionOutput[23]
        inductionTransientRate_electron_r2 = repeatSquareWaveFunctionOutput[24]
        inductionTransientRate_electron_r3 = repeatSquareWaveFunctionOutput[25]
        inductionCoverageI1 = repeatSquareWaveFunctionOutput[26]
        inductionCoverageI2 = repeatSquareWaveFunctionOutput[27]
        inductionCoverageIx = repeatSquareWaveFunctionOutput[28]
        inductionSteadyStateRate_P = repeatSquareWaveFunctionOutput[29]
        inductionSteadyStateRate_electron = repeatSquareWaveFunctionOutput[30]
        inductionSteadyStateCoverageI1 = repeatSquareWaveFunctionOutput[31]
        inductionSteadyStateCoverageI2 = repeatSquareWaveFunctionOutput[32]
        inductionSteadyStateCoverageIx = repeatSquareWaveFunctionOutput[33]
        # Let's also grab the enhancement factors here as well (without making the heat map) to check single conditions
        enhancementFactorRelativeToOscillatorySteadyState_P = repeatSquareWaveFunctionOutput[38]
        enhancementFactorRelativeToOscillatorySteadyState_electron = repeatSquareWaveFunctionOutput[39]
        enhancementFactorRelativeToOptimalSteadyState_P = repeatSquareWaveFunctionOutput[40]
        enhancementFactorRelativeToOptimalSteadyState_electron = repeatSquareWaveFunctionOutput[41]
        electrochemicalEfficiencyEnhancementFactor_P = repeatSquareWaveFunctionOutput[48]
        electrochemicalEfficiencyEnhancementFactor_electron = repeatSquareWaveFunctionOutput[49]
        inductionCycleCount = repeatSquareWaveFunctionOutput[50]
        inductionAverageDynamicRate_P = repeatSquareWaveFunctionOutput[51]
        inductionAverageDynamicRate_electron = repeatSquareWaveFunctionOutput[52]
        inductionAverageSteadyStateRate_P = repeatSquareWaveFunctionOutput[53]
        inductionAverageSteadyStateRate_electron = repeatSquareWaveFunctionOutput[54]
        inductionEnhancementFactorRelativeToSteadyState_P = repeatSquareWaveFunctionOutput[55]
        inductionEnhancementFactorRelativeToSteadyState_electron = repeatSquareWaveFunctionOutput[56]
        inductionElectrochemicalCellEfficiencyDynamic_P = repeatSquareWaveFunctionOutput[57]
        inductionElectrochemicalCellEfficiencyDynamic_electron = repeatSquareWaveFunctionOutput[58]
        inductionElectrochemicalCellEfficiencySteadyState_P = repeatSquareWaveFunctionOutput[59]
        inductionElectrochemicalCellEfficiencySteadyState_electron = repeatSquareWaveFunctionOutput[60]
   


        # Subplot #2: Binding energy or potential oscillation plot.
        if oscillationType == "Potential"
            f10 = axs[1,2].plot(dynamicSteadyStateDimensionlessTimeArray, potentialOscillation, "k-", linewidth = 3)
            axs[1,2].set_ylabel("Potential (V)", labelpad = 15, fontsize = 18)
            #axtwin = axs[1,2].twinx()
            #axtwin.plot(dynamicSteadyStateDimensionlessTimeArray, bindingEnergyOscillation, "m-", linewidth = 3)
            #axtwin.set_ylabel(L"$\Delta$H$^{0}_{I_{1}}$ (eV)", labelpad = 2, fontsize = 18)
        elseif oscillationType == "Binding Energy"
            f10 = axs[1,2].plot(dynamicSteadyStateDimensionlessTimeArray, bindingEnergyOscillation, "k-", linewidth = 3)
            axs[1,2].set_ylabel(L"$\delta$H$^{0}_{I_{1}}$ (eV)", labelpad = 2, fontsize = 18)
            #axtwin = axs[1,2].twinx()
            #axtwin.plot(dynamicSteadyStateDimensionlessTimeArray, potentialOscillation, "m-", linewidth = 3)
            #axtwin.set_ylabel("Potential (V)", fontsize = 18, labelpad= -50.0)
            #axtwin.tick_params(axis="y",labelsize=16, width=1.5)
            #axtwin.yaxis.label.set_color("magenta")
            #axtwin.set_yticks([last(potentialOscillation)])
        end
        axs[1,2].set_xlabel("Cycles", fontsize = 18)
        axs[1,2].tick_params(axis="x",labelsize=16, width=1.5)
        axs[1,2].tick_params(axis="y",labelsize=16, width=1.5)
        axs[1,2].set_title("B",loc="left",fontsize=18,fontweight="bold")
        # Subplot #3: Dynamic steady-state coverages (overlaid on oscillatory steady-state coverages)
        #f11 = axs[2,1].plot(dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateCoverageI1, "k-", dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateCoverageI2, "b-",dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateCoverageIx, "r-", dynamicSteadyStateDimensionlessTimeArray, oscillatorySteadyStateCoverageI1, "k:", dynamicSteadyStateDimensionlessTimeArray, oscillatorySteadyStateCoverageI2, "b:",dynamicSteadyStateDimensionlessTimeArray, oscillatorySteadyStateCoverageIx, "r:", linewidth = 3)
        f11 = axs[2,1].plot(dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateCoverageI1, "k-", dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateCoverageI2, "b-", dynamicSteadyStateDimensionlessTimeArray, oscillatorySteadyStateCoverageI1, "k:", dynamicSteadyStateDimensionlessTimeArray, oscillatorySteadyStateCoverageI2, "b:", linewidth = 3)
        axs[2,1].set_xlabel("Cycles", fontsize = 18)
        axs[2,1].set_ylabel("Coverage", labelpad = 5.0, fontsize = 18)
        axs[2,1].set_yticks([0,0.5,1.0])
        axs[2,1].tick_params(axis="x",labelsize=16, width=1.5)
        axs[2,1].tick_params(axis="y",labelsize=16, width=1.5)
        axs[2,1].set_title("C",loc="left",fontsize=18,fontweight="bold")
        #axs[1,3].legend(["Dynamic I1","Dynamic I2","Dynamic Ix", "Steady-State I1", "Steady-State I2", "Steady-State Ix"], loc = "upper right", fontsize=12)
        axs[2,1].legend(["I1","I2","I1 SS","I2 SS"], loc = "upper right", fontsize=10)
        # Subplot #4: Dynamic and steady-state TOF to product "P" and electrons "e-", all overlaid on each other
        f12 = axs[2,2].plot(dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateTransientRate_P, "k-", dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateTransientRate_electron, "b-", dynamicSteadyStateDimensionlessTimeArray, oscillatorySteadyStateRate_P, "k:", dynamicSteadyStateDimensionlessTimeArray, oscillatorySteadyStateRate_electron, "b:", linewidth = 3)
        axs[2,2].set_xlabel("Cycles", fontsize = 18)
        axs[2,2].set_ylabel("TOF", labelpad = 10, fontsize = 18)
        axs[2,2].set_yscale("log")
        #axs[2,1].set_ylim([-0.5,1e3])
        axs[2,2].tick_params(axis="x",labelsize=16, width=1.5)
        axs[2,2].tick_params(axis="y",labelsize=16, width=1.5)
        axs[2,2].set_title("D",loc="left",fontsize=18,fontweight="bold")
        axs[2,2].legend(["P","e-","P SS","e- SS"], loc = "lower right", fontsize=10)
        # # Subplot #4: Dynamic steady-state TOF to product "P" (overlaid on oscillatory steady-state)
        # f12 = axs[2,1].plot(dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateTransientRate_P, "k-", dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateTransientRate_electron, "k:", linewidth = 3)
        # axs[2,1].set_xlabel("Cycles", fontsize = 18)
        # axs[2,1].set_ylabel("TOF", labelpad = 10, fontsize = 18)
        # axs[2,1].set_yscale("log")
        # #axs[2,1].set_ylim([-0.5,1e3])
        # axs[2,1].tick_params(axis="x",labelsize=16, width=1.5)
        # axs[2,1].tick_params(axis="y",labelsize=16, width=1.5)
        # axs[2,1].set_title("D",loc="left",fontsize=18,fontweight="bold")
        # axs[2,1].legend(["P","e-"], loc = "upper right", fontsize=12)
        # # Subplot #5: Individual elementary step transient TOFs to e-
        # f13 = axs[2,2].plot( dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateTransientRate_electron_r1, "b-", dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateTransientRate_electron_r2, "r-", dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateTransientRate_electron_r3, "g-", linewidth = 3)
        # axs[2,2].set_xlabel("Cycles", fontsize = 18)
        # axs[2,2].set_ylabel("TOF", labelpad = 10, fontsize = 18)
        # axs[2,2].set_yscale("log")
        # #axs[2,2].set_ylim([-0.5,1e3])
        # axs[2,2].tick_params(axis="x",labelsize=16, width=1.5)
        # axs[2,2].tick_params(axis="y",labelsize=16, width=1.5)
        # axs[2,2].set_title("E",loc="left",fontsize=18,fontweight="bold")
        # axs[2,2].legend(["1","2","3"], loc = "upper right", fontsize=12)
        # # Subplot #6: blank
        # #axs[2,3].axis("off")
        # f13a = axs[2,3].plot(dynamicSteadyStateDimensionlessTimeArray, oscillatorySteadyStateRate_P, "k-", dynamicSteadyStateDimensionlessTimeArray, oscillatorySteadyStateRate_electron, "k:", linewidth = 3)
        # axs[2,3].set_xlabel("Cycles", fontsize = 18)
        # axs[2,3].set_ylabel("TOF", labelpad = 10, fontsize = 18)
        # axs[2,3].set_yscale("log")
        # #axs[2,3].set_ylim([1e-3,2e-2])
        # #axs[2,3].set_yticks([1e-3,1e-2])
        # axs[2,3].tick_params(axis="x",labelsize=16, width=1.5)
        # axs[2,3].tick_params(axis="y",labelsize=16, width=1.5)
        # axs[2,3].set_title("F",loc="left",fontsize=18,fontweight="bold")
        # axs[2,3].legend(["P","e-"], loc = "upper right", fontsize=12)
        plt.show()
        ##### MAKE THE THIRD SUBPLOT: INDUCTION PERIOD BEFORE DYNAMIC STEADY STATE OUTPUTS #####
        ################################################################################

        # Make a plot showing the dynamic TOFs overlaid on the OSS TOFs, ideally zoomed in, to compare
        # Can keep it separate from the larger subplots, since we will only show it once in the main text
        # print en
        plt.figure(figsize=(7,5))
        plt.plot(dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateTransientRate_electron_r1, "b-", dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateTransientRate_electron_r2, "r-", dynamicSteadyStateDimensionlessTimeArray, dynamicSteadyStateTransientRate_electron_r3, "g-",dynamicSteadyStateDimensionlessTimeArray, oscillatorySteadyStateRate_electron, "k:", linewidth = 3)
        plt.xlabel("Cycles", fontsize = 20)
        plt.ylabel("TOF", labelpad = 8.0, fontsize = 20)
        #plt.xlim([4.99995,5.00025])
        #plt.ylim([1e1,1e5])
        plt.yscale("log")
        #plt.xticks([5.00000,5.00025])
        #plt.yticks([1e2,1e3,1e4,1e5])
        plt.tick_params(axis="x",labelsize=20, width=1.5)
        plt.tick_params(axis="y",labelsize=20, width=1.5)
        # plt.title("A",loc="left",fontsize=18,fontweight="bold")
        plt.legend(["1","2","3"], loc = "upper right", fontsize = 16)
        plt.show() 

        if plotInductionPeriod == "yes" # makes a separate subplot of the induction coverages, TOFs, etc     
  
            # Just plot the induction coverages as a single plot instead
            plt.figure(figsize=(7.0,5.5))
            plt.plot(inductionDimensionlessTimeArray, inductionCoverageI1, "k-", inductionDimensionlessTimeArray, inductionCoverageI2, "b-", linewidth = 3)
            plt.xlabel("Cycles", fontsize = 18)
            plt.ylabel("Induction Period Coverage", labelpad = 8.0, fontsize = 18)
            # plt.yticks([0,0.5,1.0])
            plt.tick_params(axis="x",labelsize=16, width=1.5)
            plt.tick_params(axis="y",labelsize=16, width=1.5)
            # plt.title("A",loc="left",fontsize=18,fontweight="bold")
            plt.legend(["I1", "I2"], loc = "upper right", fontsize = 18)
            plt.show() 

            # inductionElectrochemicalCellEfficiencyDynamic_P = repeatSquareWaveFunctionOutput[57]
            # inductionElectrochemicalCellEfficiencyDynamic_electron = repeatSquareWaveFunctionOutput[58]
            # inductionElectrochemicalCellEfficiencySteadyState_P = repeatSquareWaveFunctionOutput[59]
            # inductionElectrochemicalCellEfficiencySteadyState_P = repeatSquareWaveFunctionOutput[60]

            inductionCycleArray = np.arange(0,inductionCycleCount+1,1)
            # Try to plot the "average TOF" and "enhancement factor" and "efficiency" terms as a function of cycle number during the induction period here
            plt.figure(figsize=(7.0,5.5))
            plt.plot(inductionCycleArray,log.(inductionAverageDynamicRate_P),"k-",inductionCycleArray,log.(inductionAverageSteadyStateRate_P),"k:",inductionCycleArray,inductionEnhancementFactorRelativeToSteadyState_P,"r-",linewidth=3)
            plt.xlabel("Induction Period Cycle Number")
            plt.ylabel("Average log TOF per cycle")
            plt.legend(["Dynamic P","Steady-state P"])

            plt.figure(figsize=(7.0,5.5))
            plt.plot(inductionCycleArray,inductionElectrochemicalCellEfficiencyDynamic_electron,"k-",inductionCycleArray,inductionElectrochemicalCellEfficiencySteadyState_electron[1]*ones(length(inductionCycleArray)),"k:",linewidth=3)
            plt.xlabel("Induction Period Cycle Number")
            plt.ylabel("Electrochemical Cell Efficiency")
            plt.legend(["Dynamic P","Steady-state P"])
            plt.ylim([0,1])
            plt.show()


        end
    end

    if heatMapsOn == "yes"
        ##### MAKE THE FOURTH SUBPLOT: ENHANCEMENT/EFFICIENCY METRIC HEAT MAPS OVER AMPLITUDE AND FREQUENCY #####
        ################################################################################
        # makeOscillationHeatMap takes in a "center" binding energy around which you can oscillate with an amplitude, or takes a "lower" potential as a lower bound of a potential oscillation (I think more intuitive this way)
        makeOscillationHeatMapOutput = makeOscillationHeatMap(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, delta_H0_I1_center, delta_H0_I1_amplitude_array, T, E_lower, E_amplitude_array, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, initialConditionFirstOverall, dutyCycle, frequencyArray, initialConditionSteadyState, numberOfDynamicSteadyStateOscillations, oscillationType, "runFullAnalysis")
        # makeOscillationHeatMap output is [enhancementFactorRelativeToOscillatorySteadyState_P_matrix, enhancementFactorRelativeToOscillatorySteadyState_electron_matrix, enhancementFactorRelativeToOptimalSteadyState_P_matrix, enhancementFactorRelativeToOptimalSteadyState_electron_matrix, turnoverEfficiency_P_matrix, turnoverEfficiency_electron_matrix, electrochemicalEfficiencyEnhancementFactor_P_matrix, electrochemicalEfficiencyEnhancementFactor_electron_matrix]
        enhancementFactorRelativeToOscillatorySteadyState_P_matrix = makeOscillationHeatMapOutput[1]
        enhancementFactorRelativeToOscillatorySteadyState_electron_matrix = makeOscillationHeatMapOutput[2]
        enhancementFactorRelativeToOptimalSteadyState_P_matrix = makeOscillationHeatMapOutput[3]
        enhancementFactorRelativeToOptimalSteadyState_electron_matrix = makeOscillationHeatMapOutput[4]
        turnoverEfficiency_P_matrix = makeOscillationHeatMapOutput[5]
        turnoverEfficiency_electron_matrix = makeOscillationHeatMapOutput[6]
        electrochemicalEfficiencyEnhancementFactor_P_matrix = makeOscillationHeatMapOutput[7]
        electrochemicalEfficiencyEnhancementFactor_electron_matrix = makeOscillationHeatMapOutput[8]
        # Make sure you plot the correct "oscillating" variable when making the heat map
        if oscillationType == "Binding Energy"
            x_axis_plot = delta_H0_I1_amplitude_array
        elseif oscillationType == "Potential"
            x_axis_plot = E_amplitude_array./2 # "E_amplitude_array" is not actually an amplitude, it is the total min-max, so need to divide by 2 to get the amplitude
        end
        # Make the appropriate meshgrid for the heatMaps
        X,Y = np.meshgrid(x_axis_plot,log10.(frequencyArray))
        ### Subplot #1: Enhancement factor with respect to oscillatory steady-state for product "P" ###
        if oscillationType == "Binding Energy"
            fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(10,4)) # 1x2 sublot if oscillating binding energy. 10x4 is a good figsize (for a 1x2 subplot) as long as you toggle tight layout when the plot shows up
            fig.tight_layout(pad=9.0) # might need to change this
            #f15 = axs[1,1].contourf(X,Y,log10.(enhancementFactorRelativeToOscillatorySteadyState_P_matrix),levels = np.linspace(0, maximum(log10.(enhancementFactorRelativeToOscillatorySteadyState_P_matrix)), 50),extend="both",cmap="viridis")
            f15 = axs[1].contourf(X,Y,log10.(enhancementFactorRelativeToOscillatorySteadyState_P_matrix),levels = np.linspace(0, 5, 50),extend="both",cmap="viridis")
            if oscillationType == "Binding Energy"
                axs[1].set_xlabel(L"$\delta$H$^{0}_{I_{1}}$ Amplitude (eV)", fontsize = 18)
            elseif oscillationType == "Potential"
                axs[1].set_xlabel("Potential Amplitude (V)", fontsize = 18)
            end
            axs[1].set_ylabel("Log Frequency", labelpad = 6.0, fontsize = 18)
            axs[1].tick_params(axis="x",labelsize=16, width=1.5)
            axs[1].tick_params(axis="y",labelsize=16, width=1.5)
            axs[1].set_title("A",loc="left",fontsize=18,fontweight="bold")
            cb15 = plt.colorbar(f15,ax=axs[1])
            #cb15.ax.set_title(label="EF_{SS_{AVG}}", pad = 12.0, fontsize = 18)
            cb15.set_ticks([0,1,2,3,4,5])
            #cb15.set_ticks(np.linspace(minimum(log10.(enhancementFactorRelativeToOscillatorySteadyState_P_matrix)),maximum(log10.(enhancementFactorRelativeToOscillatorySteadyState_P_matrix)),3)) # round these to plot the integer values instead of fractional values determined by linspace
            cb15.ax.tick_params(labelsize=14, width = 1.5)
            ### Subplot #2: Enhancement factor with respect to oscillatory steady-state for product "electron" ###
            #f16 = axs[1,2].contourf(X,Y,log10.(enhancementFactorRelativeToOscillatorySteadyState_electron_matrix),levels = np.linspace(0, maximum(log10.(enhancementFactorRelativeToOscillatorySteadyState_electron_matrix)), 50),extend="both",cmap="viridis")
            # f16 = axs[1,2].contourf(X,Y,log10.(enhancementFactorRelativeToOscillatorySteadyState_electron_matrix),levels = np.linspace(0, 4.5, 50),extend="both",cmap="viridis")
            # if oscillationType == "Binding Energy"
            #     axs[1,2].set_xlabel(L"$\Delta$H$^{0}_{I_{1}}$ Amplitude (eV)", fontsize = 18)
            # elseif oscillationType == "Potential"
            #     axs[1,2].set_xlabel("Potential Amplitude (V)", fontsize = 18)
            # end
            # axs[1,2].set_ylabel("Log Frequency", labelpad = 6.0, fontsize = 18)
            # axs[1,2].tick_params(axis="x",labelsize=16, width=1.5)
            # axs[1,2].tick_params(axis="y",labelsize=16, width=1.5)
            # axs[1,2].set_title("B",loc="left",fontsize=18,fontweight="bold")
            # cb16 = plt.colorbar(f16,ax=axs[1,2])
            # cb16.ax.set_title(label="log EF Oss. e-", pad = 12.0, fontsize = 18)
            # cb16.set_ticks([0,1,2,3,4])
            # #cb16.set_ticks(np.linspace(minimum(log10.(enhancementFactorRelativeToOscillatorySteadyState_electron_matrix)), maximum(log10.(enhancementFactorRelativeToOscillatorySteadyState_electron_matrix)), 3)) # round these to plot the integer values instead of fractional values determined by linspace
            # cb16.ax.tick_params(labelsize=14, width = 1.5)
            ### Subplot #3: Enhancement factor with respect to optimal steady-state for product "P" ###
            #f17 = axs[2,1].contourf(X,Y,log10.(enhancementFactorRelativeToOptimalSteadyState_P_matrix),levels = np.linspace(0, maximum(log10.(enhancementFactorRelativeToOptimalSteadyState_P_matrix)), 50),extend="both",cmap="viridis")
            f17 = axs[2].contourf(X,Y,log10.(enhancementFactorRelativeToOptimalSteadyState_P_matrix),levels = np.linspace(0, 2, 50),extend="both",cmap="viridis")
            if oscillationType == "Binding Energy"
                axs[2].set_xlabel(L"$\delta$H$^{0}_{I_{1}}$ Amplitude (eV)", fontsize = 18)
            elseif oscillationType == "Potential"
                axs[2].set_xlabel("Potential Amplitude (V)", fontsize = 18)
            end
            axs[2].set_ylabel("Log Frequency", labelpad = 6.0, fontsize = 18)
            axs[2].tick_params(axis="x",labelsize=16, width=1.5)
            axs[2].tick_params(axis="y",labelsize=16, width=1.5)
            axs[2].set_title("B",loc="left",fontsize=18,fontweight="bold")
            cb17 = plt.colorbar(f17,ax=axs[2])
            #cb17.ax.set_title(label="EF_{SS_{MAX}}", pad = 12.0, fontsize = 18)
            cb17.set_ticks([0,1,2])
            #cb17.set_ticks(np.linspace(minimum(log10.(enhancementFactorRelativeToOptimalSteadyState_P_matrix)), maximum(log10.(enhancementFactorRelativeToOptimalSteadyState_P_matrix)), 3)) # round these to plot the integer values instead of fractional values determined by linspace
            cb17.ax.tick_params(labelsize=14, width = 1.5)
            ### Subplot #4: Enhancement factor with respect to optimal steady-state for product "electron" ###
            #f18 = axs[2,2].contourf(X,Y,log10.(enhancementFactorRelativeToOptimalSteadyState_electron_matrix),levels = np.linspace(0, maximum(log10.(enhancementFactorRelativeToOptimalSteadyState_electron_matrix)), 50),extend="both",cmap="viridis")
            # f18 = axs[2,2].contourf(X,Y,log10.(enhancementFactorRelativeToOptimalSteadyState_electron_matrix),levels = np.linspace(0, 2, 50),extend="both",cmap="viridis")
            # if oscillationType == "Binding Energy"
            #     axs[2,2].set_xlabel(L"$\Delta$H$^{0}_{I_{1}}$ Amplitude (eV)", fontsize = 18)
            # elseif oscillationType == "Potential"
            #     axs[2,2].set_xlabel("Potential Amplitude (V)", fontsize = 18)
            # end
            # axs[2,2].set_ylabel("Log Frequency", labelpad = 6.0, fontsize = 18)
            # axs[2,2].tick_params(axis="x",labelsize=16, width=1.5)
            # axs[2,2].tick_params(axis="y",labelsize=16, width=1.5)
            # axs[2,2].set_title("D",loc="left",fontsize=18,fontweight="bold")
            # cb18 = plt.colorbar(f18,ax=axs[2,2])
            # cb18.ax.set_title(label="log EF Opt. e-", pad = 12.0, fontsize = 18)
            # cb18.set_ticks([0,1,2])
            # #cb18.set_ticks(np.linspace(minimum(log10.(enhancementFactorRelativeToOptimalSteadyState_electron_matrix)),maximum(log10.(enhancementFactorRelativeToOptimalSteadyState_electron_matrix)),3)) # round these to plot the integer values instead of fractional values determined by linspace
            # cb18.ax.tick_params(labelsize=14, width = 1.5


            plt.show()
        elseif oscillationType == "Potential"
            fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(12,4)) # if triple plot, make (12,4), if double, make (10,4), then tight layout
            fig.tight_layout(pad=9.0) # might need to change this
            #f15 = axs[1].contourf(X,Y,log10.(enhancementFactorRelativeToOscillatorySteadyState_P_matrix),levels = np.linspace(minimum(log10.(enhancementFactorRelativeToOscillatorySteadyState_P_matrix)), maximum(log10.(enhancementFactorRelativeToOscillatorySteadyState_P_matrix)), 50),extend="both",cmap="viridis")
            f15 = axs[1].contourf(X,Y,log10.(enhancementFactorRelativeToOscillatorySteadyState_P_matrix),levels = np.linspace(0.0, 3, 50),extend="both",cmap="viridis")
            if oscillationType == "Binding Energy"
                axs[1].set_xlabel(L"$\delta$H$^{0}_{I_{1}}$ Amplitude (eV)", fontsize = 18)
            elseif oscillationType == "Potential"
                axs[1].set_xlabel("Potential Amplitude (V)", fontsize = 18)
            end
            axs[1].set_ylabel("Log Frequency", labelpad = 6.0, fontsize = 18)
            axs[1].tick_params(axis="x",labelsize=16, width=1.5)
            axs[1].tick_params(axis="y",labelsize=16, width=1.5)
            axs[1].set_title("A",loc="left",fontsize=18,fontweight="bold")
            cb15 = plt.colorbar(f15,ax=axs[1])
            #cb15.ax.set_title(label="log EF Oss. P", pad = 12.0, fontsize = 18)
            cb15.set_ticks([0,1,2,3])
            #cb15.set_ticks(np.linspace(minimum(log10.(enhancementFactorRelativeToOscillatorySteadyState_P_matrix)),maximum(log10.(enhancementFactorRelativeToOscillatorySteadyState_P_matrix)),3)) # round these to plot the integer values instead of fractional values determined by linspace
            cb15.ax.tick_params(labelsize=14, width = 1.5)
            ### Subplot #2: Enhancement factor with respect to oscillatory steady-state for product "electron" ###
            # f16 = axs[1,2].contourf(X,Y,log10.(enhancementFactorRelativeToOscillatorySteadyState_electron_matrix),levels = np.linspace(minimum(log10.(enhancementFactorRelativeToOscillatorySteadyState_electron_matrix)),maximum(log10.(enhancementFactorRelativeToOscillatorySteadyState_electron_matrix)),50),extend="both",cmap="viridis")
            # if oscillationType == "Binding Energy"
            #     axs[1,2].set_xlabel(L"$\Delta$H$^{0}_{I_{1}}$ Amplitude (eV)", fontsize = 18)
            # elseif oscillationType == "Potential"
            #     axs[1,2].set_xlabel("Potential Amplitude (V)", fontsize = 18)
            # end
            # axs[1,2].set_ylabel("Log Frequency", labelpad = 6.0, fontsize = 18)
            # axs[1,2].tick_params(axis="x",labelsize=16, width=1.5)
            # axs[1,2].tick_params(axis="y",labelsize=16, width=1.5)
            # axs[1,2].set_title("B",loc="left",fontsize=18,fontweight="bold")
            # cb16 = plt.colorbar(f16,ax=axs[1,2])
            # cb16.ax.set_title(label="log EF Oss. e-", pad = 12.0, fontsize = 18)
            # cb16.set_ticks([0,0.05,0.1,0.15])
            # #cb16.set_ticks(np.linspace(minimum(log10.(enhancementFactorRelativeToOscillatorySteadyState_electron_matrix)), maximum(log10.(enhancementFactorRelativeToOscillatorySteadyState_electron_matrix)), 3)) # round these to plot the integer values instead of fractional values determined by linspace
            # cb16.ax.tick_params(labelsize=14, width = 1.5)
            ### Subplot #3: Enhancement factor with respect to optimal steady-state for product "P" ###
            #f17 = axs[2].contourf(X,Y,log10.(enhancementFactorRelativeToOptimalSteadyState_P_matrix),levels = np.linspace(minimum(log10.(enhancementFactorRelativeToOptimalSteadyState_P_matrix)), maximum(log10.(enhancementFactorRelativeToOptimalSteadyState_P_matrix)), 50),extend="both",cmap="viridis")
            f17 = axs[2].contourf(X,Y,log10.(enhancementFactorRelativeToOptimalSteadyState_P_matrix),levels = np.linspace(-1.5,1.5, 50),extend="both",cmap="viridis")
            if oscillationType == "Binding Energy"
                axs[2].set_xlabel(L"$\delta$H$^{0}_{I_{1}}$ Amplitude (eV)", fontsize = 18)
            elseif oscillationType == "Potential"
                axs[2].set_xlabel("Potential Amplitude (V)", fontsize = 18)
            end
            axs[2].set_ylabel("Log Frequency", labelpad = 6.0, fontsize = 18)
            axs[2].tick_params(axis="x",labelsize=16, width=1.5)
            axs[2].tick_params(axis="y",labelsize=16, width=1.5)
            axs[2].set_title("B",loc="left",fontsize=18,fontweight="bold")
            cb17 = plt.colorbar(f17,ax=axs[2])
            #cb17.ax.set_title(label="log EF Opt. P", pad = 12.0, fontsize = 18)
            cb17.set_ticks([-1.5,0,1.5])
            #cb17.set_ticks(np.linspace(minimum(log10.(enhancementFactorRelativeToOptimalSteadyState_P_matrix)), maximum(log10.(enhancementFactorRelativeToOptimalSteadyState_P_matrix)), 3)) # round these to plot the integer values instead of fractional values determined by linspace
            cb17.ax.tick_params(labelsize=14, width = 1.5)
            ### Subplot #4: Enhancement factor with respect to optimal steady-state for product "electron" ###
            # f18 = axs[2,1].contourf(X,Y,log10.(enhancementFactorRelativeToOptimalSteadyState_electron_matrix),levels = np.linspace(minimum(log10.(enhancementFactorRelativeToOptimalSteadyState_electron_matrix)), maximum(log10.(enhancementFactorRelativeToOptimalSteadyState_electron_matrix)), 50),extend="both",cmap="viridis")
            # if oscillationType == "Binding Energy"
            #     axs[2,1].set_xlabel(L"$\Delta$H$^{0}_{I_{1}}$ Amplitude (eV)", fontsize = 18)
            # elseif oscillationType == "Potential"
            #     axs[2,1].set_xlabel("Potential Amplitude (V)", fontsize = 18)
            # end
            # axs[2,1].set_ylabel("Log Frequency", labelpad = 6.0, fontsize = 18)
            # axs[2,1].tick_params(axis="x",labelsize=16, width=1.5)
            # axs[2,1].tick_params(axis="y",labelsize=16, width=1.5)
            # axs[2,1].set_title("D",loc="left",fontsize=18,fontweight="bold")
            # cb18 = plt.colorbar(f18,ax=axs[2,1])
            # cb18.ax.set_title(label="log EF Opt. e-", pad = 12.0, fontsize = 18)
            # cb18.set_ticks([-0.3,-0.2,-0.1,0])
            # #cb18.set_ticks(np.linspace(minimum(log10.(enhancementFactorRelativeToOptimalSteadyState_electron_matrix)),maximum(log10.(enhancementFactorRelativeToOptimalSteadyState_electron_matrix)),3)) # round these to plot the integer values instead of fractional values determined by linspace
            # cb18.ax.tick_params(labelsize=14, width = 1.5)
            ### Subplot #5: Electrochemical efficiency enhancement for product "P" ###
            # f19 = axs[2,2].contourf(X,Y,100 .*electrochemicalEfficiencyEnhancementFactor_P_matrix,levels=np.linspace(-5,5,50),extend="both",cmap="viridis") 
            # if oscillationType == "Binding Energy"
            #     axs[2,2].set_xlabel(L"$\Delta$H$^{0}_{I_{1}}$ Amplitude (eV)", fontsize = 18)
            # elseif oscillationType == "Potential"
            #     axs[2,2].set_xlabel("Potential Amplitude (V)", fontsize = 18)
            # end
            # axs[2,2].set_ylabel("Log Frequency", labelpad = 6.0, fontsize = 18)
            # axs[2,2].tick_params(axis="x",labelsize=16, width=1.5)
            # axs[2,2].tick_params(axis="y",labelsize=16, width=1.5)
            # axs[2,2].set_title("E",loc="left",fontsize=18,fontweight="bold")
            # cb19 = plt.colorbar(f19,ax=axs[2,2])
            # cb19.ax.set_title(label="EEF P", pad = 12.0, fontsize = 18)
            # cb19.set_ticks([0,5,10,15,20])
            # cb19.ax.tick_params(labelsize=14, width = 1.5)
            ### Subplot #6: Electrochemical efficiency enhancement for product "electron" ###
            f20 = axs[3].contourf(X,Y,100 .*electrochemicalEfficiencyEnhancementFactor_electron_matrix,levels=np.linspace(-50,0,50),extend="both",cmap="viridis") # get percentages for efficiency
            if oscillationType == "Binding Energy"
                axs[3].set_xlabel(L"$\delta$H$^{0}_{I_{1}}$ Amplitude (eV)", fontsize = 18)
            elseif oscillationType == "Potential"
                axs[3].set_xlabel("Potential Amplitude (V)", fontsize = 18)
            end
            axs[3].set_ylabel("Log Frequency", labelpad = 6.0, fontsize = 18)
            axs[3].tick_params(axis="x",labelsize=16, width=1.5)
            axs[3].tick_params(axis="y",labelsize=16, width=1.5)
            axs[3].set_title("C",loc="left",fontsize=18,fontweight="bold")
            cb20 = plt.colorbar(f20,ax=axs[3])
            #cb20.ax.set_title(label="EEF e-", pad = 12.0, fontsize = 18)
            cb20.set_ticks([-50,-25,0])
            cb20.ax.tick_params(labelsize=14, width = 1.5)
            plt.show()
        end
        
    end
    return enhancementFactorRelativeToOscillatorySteadyState_P_matrix, enhancementFactorRelativeToOptimalSteadyState_P_matrix, electrochemicalEfficiencyEnhancementFactor_electron_matrix
end


# BINDING ENERGY, SYMMETRICAL, 3.1.1 SECTION
# runFullAnalysis(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, [-0.25,0.25], T, [0.1,0.1], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, [0.33,0.33,0.00], 50, 0.1, [0.33,0.33,0.00], 1, "Binding Energy", "P", "yes", "yes", "yes", [1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3], 0, [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45], 0.1, [0,0,0,0,0,0,0,0], "no", "yes")

# Try just the frequency plots
# This will be the actual induction periods
#plt.figure(1)
# frequencyArray = [1e0,2.5e0,5e0,7.5e0,1e1,2.5e1,5e1,7.5e1,1e2,2.5e2,5e2,7.5e2,1e3,2.5e3,5e3,7.5e3,1e4]
# freqOutput = varyFrequencyConstantAmplitude(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, 0, 0.35, T, 0.1, 0.0, gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, [0.5,0.5,0.00], 50, frequencyArray, [0.33,0.33,0.00], 1, "Binding Energy", "runFullAnalysis")

# enhancementFactorRelativeToOscillatorySteadyState_P_array = freqOutput[1]
# enhancementFactorRelativeToOptimalSteadyState_P_array = freqOutput[2]
# # This will be just the plot of enhancement terms vs. freq.
# plt.figure(2)
    
# plt.plot(log10.(frequencyArray), log10.(enhancementFactorRelativeToOscillatorySteadyState_P_array), "k-",log10.(frequencyArray), log10.(enhancementFactorRelativeToOptimalSteadyState_P_array), "r-")
# plt.xlabel("Log10 Frequency")
# plt.ylabel("Log10 Enhancement Factor")
# plt.legend(["Average","Optimal"])
# plt.show()

# POTENTIAL, LESS POINTS IN HEAT MAPS, SYMMETRICAL, 3.1.2 SECTION
# fullAnalysisOutputs = runFullAnalysis(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, [0,0], T, [0.1,0.5], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, [0.33,0.33,0.00], 50, 0.01, [0.33,0.33,0.00], 1, "Potential", "P", "yes", "yes", "no", [1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3], -0.3, [0,0,0,0,0,0,0,0], 0.1, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], "no", "yes")
# POTENTIAL, LESS POINTS IN HEAT MAPS, ASYMMETRICAL 1, 3.1.2 SECTION
fullAnalysisOutputs = runFullAnalysis(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, [-0.25,-0.25], T, [0.1,0.5], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, [0.33,0.33,0.00], 50, 1, [0.33,0.33,0.00], 1, "Potential", "P", "yes", "yes", "yes", [1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3], -0.25, [0,0,0,0,0,0,0,0], 0.1, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], "no", "yes")
# POTENTIAL, LESS POINTS IN HEAT MAPS, SYMMETRICAL, 3.2.1 - POISONING INTERMEDIATE 
# fullAnalysisOutputs = runFullAnalysis(H0_array_Mref_Eref, S0_array_Mref_Eref, H0_poisoning_array_Mref_Eref, S0_poisoning_array_Mref_Eref, [0.0,0.0], T, [0.1,0.5], gamma_BEPs, gamma_BEP_poison, del_scaling_params, del_scaling_param_poison, betas, beta_poison, [0.33,0.33,0.50], 50, 10, [0.33,0.33,0.50], 1, "Potential", "P", "yes", "yes", "yes", [1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3], 0, [0,0,0,0,0,0,0,0], 0.1, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8], "yes", "yes")

