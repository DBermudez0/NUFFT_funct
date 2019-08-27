# NUFFT_funct
Non-uniform Fourier Transform for radial data reconstruction in Python
<h3> Python based NUFFT (based of Jeff Fessler NUFFT MATLAB package)</h3>
  <img src ="Full_sampled_liver_recon.png" width="800" height="600" />
  
  <img src="Full_sampled(phantom).png" width="800" height="600" />


<h4> functions withing the NUFFT_funct.py script: </h4>
<h5> MCNUFFT:</h5> function to generate the interpolation sparcity matrix and initializing all the initial values for reconstruction (used for multi-coil multi-channel data)
<h5> nufft_adj:</h5> function used to go from k-space doamin to image space.
<h5> nufft:</h5> function used to go from image space to k-space.

<h4> functions within the mtimes_funct.py script: </h4>
<h5> mtimes: </h5> function to go from k-sapce to image space (or vise-verse) for multi-coil multi-channel data.
