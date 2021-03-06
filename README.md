# NUFFT_funct
Non-uniform Fourier Transform for radial data reconstruction in Python
<h3> Python based NUFFT (based of Jeff Fessler NUFFT MATLAB package)</h3>
  <img src ="Full_sampled_liver_recon.png" width="800" height="600" />
  
  <img src="Full_sampled(phantom).png" width="800" height="600" />


<h4> Functions withing the NUFFT_funct.py script: </h4>
<h5> MCNUFFT:</h5> function to generate the interpolation sparcity matrix and initializing all the initial values for reconstruction (used for multi-coil multi-channel data)
<h5> nufft_adj:</h5> function used to go from k-space to image space.
<h5> nufft:</h5> function used to go from image space to k-space.

<h4> Functions within the mtimes_funct.py script: </h4>
<h5> mtimes:</h5> function to go from k-space to image space (or vise-verse) for multi-coil multi-channel data.

<h4> Example </h4>
run demo_liver_recon.py to see how to reconstruct a radial liver MRI dataset for a single-coil single channel data.

<h4> Requirements </h4>
scipy, numpy, matplotlib, math, cmath, and mpmath (https://github.com/fredrik-johansson/mpmath) modules.

<h4> Instructions </h4>

(Instructions if you do not have mpmath installed)
<p> 1. type in command line git clone https://github.com/fredrik-johansson/mpmath </p>
<p> 2. cd mpmath </p>
<p> 3. type in command line git clone https://github.com/DBermudez0/NUFFT_funct </p>
<p> 4. cd NUFFT_funct </p>
<p> 5. type in command line mv * ../ </p>
<p> 6. cd .. </p>
<p> 5. type in command line: python demo_liver_recon.py </p>

(If you have already a mpmath module installed in a directory within your PYTHONPATH)
<p> 1. type in command line git clone https://github.com/DBermudez0/NUFFT_funct </p>
<p> 2. execute python demo_liver_recon.py </p>
