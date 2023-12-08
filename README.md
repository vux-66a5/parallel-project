# Practical single-shot quantitative phase imaging with in-line holography
_______
<p align="left">
<img src="diagram.png">
</p>

<p align="left"> <strong>Figure 1</strong>. Overview of the proposed method. (a) Schematic of the in-line holographic imaging system. (b) Captured raw hologram of a transparent Fresnel zone plate. Scale bar 1 mm. (c) Retrieved phase distribution. (d) Rendered surface height profile.</p>

## Requirements
Install libary by using `pip instal -r requirements.txt`

## Quick Start
- **Phase retrieval using simulated data.** Run `demo_sim.m` with default parameters.
- **Phase retrieval using experimental data.
- **Try on your own experiment data.** Prepare a hologram and an optional reference image, run `preprocessing.m` and set the experiment parameters (e.g. pixel size, wavelength, and sample-to-sensor distance). Then run `demo_exp.m` and see how it works.
