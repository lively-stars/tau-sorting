Then one sorts all the points into those tau- groups.
Knowing all the sub-bins that go into a particular tau-group, one needs to calculate the Planck mean and the Rosseland mean with the corresponding opacities accouting for the delta lambda. That is for example, if you have a sub-bin with width 0.1 in the bin with delta lambda 2nm, the delta lambda for that is 0.1*2nm.

You can check out either Tanayveer's tausort implentation or the tausort.c for the formula for the Planck and Rosseland mean, OR, to the paper https://www.aanda.org/articles/aa/pdf/2004/26/aa0043.pdf The equations 6, 11, 12

As for now the mean opacities in each group is calculated using the equation 12 in that paper with the threshold 0.35 to switch between planck and rosseland mean (as Ludwig suggested).
