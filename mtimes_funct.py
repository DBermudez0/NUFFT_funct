def mtimes(a,bb):
    # a is the structure obtained through the function MCNUFFT
    # bb is either the multi-variate k-space data or the image data in the image domain
    import numpy as np
    from NUFFT_funct import nufft_adj, nufft

    if a["adjoint"]==0:
        #Multicoil non-Cartesian k-space to Cartesian image domain nufft for each coil and time point
        res = np.zeros((a["imSize"][0],a["imSize"][1],np.shape(bb)[2]), dtype = np.complex128)
        
        ress = np.zeros((a["imSize"][0],a["imSize"][1], np.shape(bb)[3]),dtype = np.complex128)
        
        for tt in range(0, np.shape(bb)[3]):
            for ch in range(0, np.shape(bb)[2]):
                    b = bb[:,:,ch,tt]*a["w"][:,:,tt]
                    b = np.reshape(np.swapaxes(b,0,1), (np.shape(b)[0]*np.shape(b)[1],1))
                    
                    
                    res[:,:,ch] = np.reshape(nufft_adj(b,a["st"][tt])/np.sqrt(np.prod(a["imSize2"])), (a["imSize"][0], a["imSize"][1]))
                    if ch == 2:
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt
                        plt.imshow(np.abs(res[:,:,ch]),cmap="gray")
                        plt.savefig("NUFFT_recon_test.png")

            ress[:,:,tt] = np.sum(res*np.conj(a["b1"]), axis=2)/np.sum(np.abs((np.squeeze(a["b1"])))**2,axis=2)
        
        ress = ress*np.shape(a["w"])[0]*math.pi/2/np.shape(a["w"])[1]
        
    else:
        print(np.shape(bb))
        res = np.zeros((a["imSize"][0],a["imSize"][1]), dtype = np.complex128)
        ress = np.zeros((a["dataSize"][0],a["dataSize"][1],np.shape(a["b1"])[2],np.shape(bb)[2]), dtype = np.complex128)
        print(np.shape(ress))
                
        for tt in range(0,np.shape(bb)[2]):
            for ch in range(0,np.shape(a["b1"])[2]):
                res = bb[:,:,tt]*a["b1"][:,:,ch]
                ress[:,:,ch,tt] = np.reshape(nufft(res,a["st"][tt])/np.sqrt(np.prod(a["imSize2"])), (a["dataSize"][0],a["dataSize"][1]))*a["w"][:,:,tt]
    return ress



def mtimes_2(a,b):
    # Radial k-space <-> image sample ordering: the sparse matrix p was built
    # from the trajectory flattened as swapaxes(.,0,1).ravel() (see MCNUFFT).
    # BOTH branches below must use that SAME 2D<->vector map so the forward and
    # adjoint are consistent; otherwise the samples are scrambled and the CS
    # solve collapses to noise.
    def kspace2vec(b2d):                       # (nx, nline) -> (M,) vector
        return np.reshape(np.swapaxes(b2d, 0, 1), (b2d.shape[0] * b2d.shape[1],))
    def vec2kspace(v, nx, nline):              # (M,) -> (nx, nline)
        return np.swapaxes(np.reshape(v, (nline, nx)), 0, 1)

    if a["adjoint"] == 0:
        # ---- ADJOINT: multicoil non-Cartesian k-space -> image ----
        res = np.zeros((a["imSize"][0], a["imSize"][1], np.shape(bb)[2]),
                       dtype=np.complex128)
        ress = np.zeros((a["imSize"][0], a["imSize"][1], np.shape(bb)[3]),
                        dtype=np.complex128)
        for tt in range(0, np.shape(bb)[3]):
            for ch in range(0, np.shape(bb)[2]):
                b = bb[:, :, ch, tt] * a["w"][:, :, tt]     # density comp (sqrt w)
                bvec = kspace2vec(b)
                img = nufft_adj(bvec, a["st"][tt]) / np.sqrt(np.prod(a["imSize2"]))
                res[:, :, ch] = np.reshape(img, (a["imSize"][0], a["imSize"][1]))
            # coil combination: adjoint applies conj(b1) and SUMS over coils.
            # This is the exact transpose of the forward's multiply-by-b1 step.
            # (Do NOT divide by sum|b1|^2 here: that gridding normalization
            #  would make E' not the true adjoint of E, breaking CG. Assumes
            #  b1 are normalized so sum|b1|^2 ~ 1, as in the MCNUFFT setup.)
            ress[:, :, tt] = np.sum(res * np.conj(a["b1"]), axis=2)
        # NOTE: no extra adjoint-only scaling here. Any constant applied only in
        # the adjoint (e.g. the old nx*pi/2/nline gridding factor) makes E' not
        # the exact transpose of E, so E'E is no longer Hermitian and CG can
        # stall/diverge. Density compensation is already applied symmetrically
        # as sqrt(w) in both directions.

    else:
        # ---- FORWARD: image -> multicoil non-Cartesian k-space ----
        ress = np.zeros((a["dataSize"][0], a["dataSize"][1],
                         np.shape(a["b1"])[2], np.shape(bb)[2]), dtype=np.complex128)
        nx, nline = a["dataSize"][0], a["dataSize"][1]
        for tt in range(0, np.shape(bb)[2]):
            for ch in range(0, np.shape(a["b1"])[2]):
                res = bb[:, :, tt] * a["b1"][:, :, ch]      # apply coil sensitivity
                Xvec = nufft(res, a["st"][tt]) / np.sqrt(np.prod(a["imSize2"]))
                Xf = vec2kspace(Xvec, nx, nline)            # (nx, nline), mirrors adjoint
                ress[:, :, ch, tt] = Xf * a["w"][:, :, tt]  # density comp (sqrt w)
    return ress
    return ress

