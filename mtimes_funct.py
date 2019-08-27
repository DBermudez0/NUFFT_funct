def mtimes(a,bb):
    import numpy as np
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


