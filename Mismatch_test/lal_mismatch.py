import numpy as np
import os
import math
import h5py
import scipy
from scipy import optimize
from scipy.optimize import minimize
import sys
import time

import lalsimulation as ls
import lal

from scipy.spatial.transform import Rotation as R

os.environ['LAL_DATA_PATH'] = '/home/alvaroc/miniconda3/envs/lal/share/lalsimulation'


# optimization wrapper

def minimize_wrapper(func, IG_list, optimization_method):
    """
        Minimize a ndim-dimensional function.
        Needs scipy version >= 0.19.1

        Inputs:
        func:               Function to minimize. Takes ndim list/array as input
        and returns a float.
        IG_list:            List of initial guesses. Should be a list/array of
        lists/arrays. The inner list/array should be ndim
        dimensional.
        Returns:
        x, func(x) at minima.
        """

    # For each initial guess, run minimization and finally pick the min among those results

    x_vec = []
    minFuncVal_vec = []
    for i in range(len(IG_list)):
        if(optimization_method=='NM'):
           res = scipy.optimize.minimize(func, IG_list[i], method='Nelder-Mead',options={'disp': False, 'fatol': 1e-4})
        else:
           res = scipy.optimize.minimize(func, IG_list[i], method='SLSQP',bounds=((0.,2.*np.pi),(0.,2.*np.pi)),options={'eps': 0.05, 'disp': False, 'ftol': 1e-3})
        x_vec.append(res.x)
        minFuncVal_vec.append(res.fun)

    minIdx = np.argmin(minFuncVal_vec)
    return x_vec[minIdx], minFuncVal_vec[minIdx]


def dual_annealing(func,bounds,maxfun=500):

    result= optimize.dual_annealing(func, bounds, maxfun=maxfun, minimizer_kwargs={'method': 'SLSQP','tol': 1e-3})
    opt_pars,opt_val=result['x'],result['fun']

    return opt_pars, opt_val



# rotate in-plane spins --used for Phenom models
def RotateSpinVectors(S1,S2,alpha=0.):

    r = R.from_rotvec(alpha * np.array([0, 0, 1]))
    S1_new=r.apply(S1)
    S2_new=r.apply(S2)

    return S1_new, S2_new


## lal mismatch tools (fixed)

def pol_response(hp, hc, pol):
    '''Combines h_+ and h_x returned from Choose*Waveform into a detector response,
        h_res = cos(2 psi) h_+ + sin(2 psi) h_x'''

    return np.cos(2.*pol) * hp.data.data + np.sin(2.*pol) * hc.data.data


def psdarray(psd, flow, fhigh, frequencies):
    '''Returns a numpy array containing the PSD values at frequencies,
       where PSD = 1 for f < flow and f > fhigh.'''
    inband = (frequencies >= flow) & (frequencies <= fhigh)
    Sn = np.ones_like(frequencies)
    Sn[inband] = np.fromiter(map(psd, frequencies[inband]),dtype=float)
    return Sn


def optSNR(h, psd, flow, fhigh, frequencies=None):
    '''Calculates the optimal SNR of a frequency-domain signal.'''

    if frequencies is None: #h is output for Choose*Waveform
        freqs = np.arange(h.data.data.size) * h.deltaF
        hdata = h.data.data.copy()
    else: #h and frequencies are NumPy arrays
        freqs = frequencies
        hdata = h.copy()
        assert h.size == freqs.size, 'Frequency and signal array sizes do not match!'

    outofband = (freqs < flow) | (freqs > fhigh)

    if isinstance(psd, np.ndarray): #psd is already the appropriate array
        assert psd.size == freqs.size, 'Frequency and PSD array sizes do not match!'
        Sn = psd
    else:
        Sn = psdarray(psd, flow, fhigh, freqs)

    hdata[outofband] = 0. + 0.j
    integrand = np.abs(hdata)**2 / Sn
    # df = freqs[1] - freqs[0]
    return np.sqrt(np.sum(integrand))


def pol_opt_match(signal, template, psd, flow, fhigh, signal_norm=None, signal_polarisation=0.,
                  deltaF=None, zpfact=8):
    '''Calculates the match between signal and template, both given in their frequency-domain
       representation.'''

    if signal_polarisation is None: #case for signal being already a detector response
        assert deltaF > 0, 'Positive deltaF required when no polarisation given.'
        hs_resp = signal.copy()
        df = deltaF
    else:
        hsp, hsc = signal
        df = hsp.deltaF
        hs_resp = pol_response(hsp, hsc, signal_polarisation)

    htp, htc = template
    assert df == htp.deltaF, 'Signal and template have different deltaF.'

    maxlength = max(hs_resp.size, htp.data.data.size)
    freqs =  np.arange(maxlength) * df

    if isinstance(psd, np.ndarray): #psd is already the appropriate array
        assert psd.size >= maxlength, 'The psd array is not long enough. Does it contain all frequencies?'
        Sn = psd
    else:
        Sn = psdarray(psd, flow, fhigh, freqs)

    if signal_norm is None:
        norm_s = optSNR(hs_resp, Sn[:hs_resp.size], flow, fhigh, freqs[:hs_resp.size])
    else:
        norm_s = signal_norm

    norm_tp = optSNR(htp, Sn[:htp.data.data.size], flow, fhigh)
    norm_tc = optSNR(htc, Sn[:htc.data.data.size], flow, fhigh)

    #assert (norm_tp > 0 and norm_tc > 0), 'Plus or cross polarisation of the template vanish.'

    freqs  = freqs[:htp.data.data.size]
    outofband = (freqs < flow) | (freqs > fhigh)

    if(norm_tp > 0 and norm_tc > 0):
        htp_data = htp.data.data.copy() / norm_tp
        htc_data = htc.data.data.copy() / norm_tc
    else:
        raise RuntimeError



    htp_data[outofband] = 0. + 0.j
    htc_data[outofband] = 0. + 0.j

    Ipc = np.real(np.dot(np.conjugate(htp_data), htc_data/Sn[:htp_data.size]))

    if htp_data.size > hs_resp.size:
        htp_data = htp_data[:hs_resp.size]
        htc_data = htc_data[:hs_resp.size]
        dataL = hs_resp.size
    else:
        hs_resp = hs_resp[:htp_data.size]
        dataL = htp_data.size

    Sn = Sn[:dataL]
    fftlen = 2**(int(np.log(zpfact*dataL)/np.log(2))+1)
    rho_p = (fftlen)*np.fft.ifft(np.conjugate(htp_data)*hs_resp/Sn, n=fftlen)
    rho_c = (fftlen)*np.fft.ifft(np.conjugate(htc_data)*hs_resp/Sn, n=fftlen)
    gamma = np.real(rho_p*np.conjugate(rho_c))

    rho_p2 = np.abs(rho_p)**2
    rho_c2 = np.abs(rho_c)**2

    sqrt_part = np.sqrt((rho_p2-rho_c2)**2 + 4*(Ipc*rho_p2-gamma)*(Ipc*rho_c2-gamma))
    num = rho_p2 - 2.*Ipc*gamma + rho_c2 + sqrt_part
    den = 1. - Ipc**2
    overlap = np.sqrt(np.max(num)/den/2.)/norm_s
    return overlap



def pol_phase_optimized_match(signal, template_pars, psd, flow, fhigh,
                              signalpol, rotate_spins=0, optimization_method='NM', zpfact=2,
                              test_phi = np.linspace(0.,2*np.pi, 4, endpoint=False), alphas=np.linspace(0.,2*np.pi, 4, endpoint=False),maxfun=500):
    '''Calculates the match between a signal and template, optimised over the template's
       polarisation and orbital phase.

       Input:
       ======
       signal: FD numpy array
       template_pars: dictionary containing template parameters suitable for SimInspiralFD
       psd: analytica function of the PSD
       flow, fhigh: inner produce cutoff frequencies (in Hz)
       signalpol: signal polarisation
       optimisation_method: 'NM' (Nelder Mead), 'discrete' (over array), 'none'
       zpfact: padding factor to increase time-resolution [default: 2]
       test_phi: array of orbital phases for 'discrete' optimisation (or initial
                 guesses for 'NM')  [default: []]


       Output:
       ======
       [match, signal norm, phase value]
       '''
    hsp, hsc = signal

    #Store spins of NR signal
    S1=[template_pars['S1x'],template_pars['S1y'],template_pars['S1z']]
    S2=[template_pars['S2x'],template_pars['S2y'],template_pars['S2z']]

    # Check both waveforms have been generated with the same deltaF
    assert hsp.deltaF == template_pars['deltaF'], "Template and signal have different Delta_f."

    hs_res = pol_response(hsp, hsc, signalpol)

    template_length = int((template_pars['f_max'] / template_pars['deltaF'])) + 1
    maxlength = max(template_length, hs_res.size)
    frequencies = np.arange(maxlength) * hsp.deltaF

    Sn = psdarray(psd, flow, fhigh, frequencies)

    hs_res_norm = optSNR(hs_res, Sn[:hs_res.size], flow, fhigh, frequencies[:hs_res.size])
    pars = template_pars.copy()


    def mylocalf(x):
        phi_test=x[0]
        alpha=x[1]
        S1_new,S2_new=RotateSpinVectors(S1,S2,alpha=alpha)
        pars.update({'S1x': S1_new[0]}); pars.update({'S1y': S1_new[1]});
        pars.update({'S2x': S2_new[0]}); pars.update({'S2y': S2_new[1]});
        pars.update({'phiRef': phi_test})
        try:
            template = ls.SimInspiralFD(**pars)
        except RuntimeError:
            raise
        return 1.-pol_opt_match(hs_res, template, Sn, flow, fhigh, signal_polarisation=None, deltaF=hsp.deltaF,
                            signal_norm=hs_res_norm, zpfact=zpfact)
    
    
    def mylocalf3D(x):
        phi_test=x[0]
        alpha=x[1]
        beta=x[2]
        r = R.from_rotvec(alpha * np.array([0, 0, 1]))
        S1_new=r.apply(S1)
        r = R.from_rotvec(beta * np.array([0, 0, 1]))
        S2_new=r.apply(S2)
        pars.update({'S1x': S1_new[0]}); pars.update({'S1y': S1_new[1]});
        pars.update({'S2x': S2_new[0]}); pars.update({'S2y': S2_new[1]});
        pars.update({'phiRef': phi_test})
        try:
            template = ls.SimInspiralFD(**pars)
        except RuntimeError:
            raise
        return 1.-pol_opt_match(hs_res, template, Sn, flow, fhigh, signal_polarisation=None, deltaF=hsp.deltaF,
                                signal_norm=hs_res_norm, zpfact=zpfact)


    if(len(test_phi)==0):
        test_phi=[0.]

    if rotate_spins==0:
        iniguess=np.stack((test_phi, np.repeat(0.,len(test_phi))), axis=-1)
    else:
        assert len(alphas) > 0, 'No set of spin-angles given. Please set alphas!'
        iniguess=np.stack((test_phi,np.repeat(alphas[0],len(test_phi))),axis=-1)
        for alpha in alphas[1:len(alphas)]:
            iniguess=np.concatenate((np.stack((test_phi,np.repeat(alpha,len(test_phi))),axis=-1),iniguess))


    if (optimization_method=='NM') or (optimization_method=='SLSQP'):
        try:
            opt_pars, mm = minimize_wrapper(mylocalf, iniguess, optimization_method)
        except RuntimeError:
            raise
        return [opt_pars, hs_res_norm, 1-mm]
    elif optimization_method=='dual_a':
        try:
            opt_pars,mm=dual_annealing(mylocalf3D,((0.,2.*np.pi),(0.,2.*np.pi),(0.,2.*np.pi)),maxfun=maxfun)
        except RuntimeError:
            raise
        return [opt_pars, hs_res_norm, 1-mm]

    elif optimization_method=='discrete':
        assert len(test_phi) > 0, 'No set of angles given. Please set test_phi!'
        try:
            match_vals = np.array([1.-mylocalf(phi) for phi in iniguess])
        except RuntimeError:
            raise
        return [iniguess[np.argmax(match_vals)], hs_res_norm, np.max(match_vals)]
    elif optimization_method=='none':
        if len(test_phi) > 0:
            try:
                match_vals = np.array([1.-mylocalf(phi) for phi in iniguess])
            except RuntimeError:
                raise
            return np.array([iniguess[:,0],iniguess[:,1],match_vals]).T
        else:
            try:
                return [[template_pars['phiRef'],0.],hs_res_norm,1.-mylocalf([template_pars['phiRef'],0.])]
            except RuntimeError:
                raise


#______________________________________________ NR and template generation functions _______________________________________________________

def GenerateNRsignal(filepath, mtotal, inclination, delta_F=0.125, buffer=1.,params = lal.CreateDict(),distance = 100.*1e6*lal.PC_SI, phiRef = 0.,modes=[]):


    f = h5py.File(filepath, 'r')

    if(len(modes)>0):

        modearray= ls.SimInspiralCreateModeArray()
        for mode in modes:
            ls.SimInspiralModeArrayActivateMode(modearray, mode[0], mode[1])
        ls.SimInspiralWaveformParamsInsertModeArray(params, modearray)


    ls.SimInspiralWaveformParamsInsertNumRelData(params, filepath)

    m1 = f.attrs['mass1']
    m2 = f.attrs['mass2']

    m1SI = m1 * mtotal / (m1 + m2) * lal.MSUN_SI
    m2SI = m2 * mtotal / (m1 + m2) * lal.MSUN_SI

    f_lower = f.attrs['f_lower_at_1MSUN']/mtotal
    f_lower= buffer*f_lower

    fRef=f_lower
    #print(fRef)

    #f_lower = 15.
    #fRef = 15.
    
    # The NR spins need to be transformed into the lal frame:
    # spins = ls.SimInspiralNRWaveformGetSpinsFromHDF5File(fRef, mtotal, filepath)
    spins = ls.SimInspiralNRWaveformGetSpinsFromHDF5File(-1, 1, filepath) 

    spin1x = spins[0]
    spin1y = spins[1]
    spin1z = spins[2]
    spin2x = spins[3]
    spin2y = spins[4]
    spin2z = spins[5]

    f.close()


    f_max = 2**(int(np.log(3000./delta_F)/np.log(2))+1) * delta_F
    inspiralFDparams = {
    'm1': m1SI, 'm2': m2SI,
    'S1x': spin1x, 'S1y': spin1y, 'S1z': spin1z,
    'S2x': spin2x, 'S2y': spin2y, 'S2z': spin2z,
    'distance': distance, 'inclination': inclination,
    'phiRef': phiRef, 'longAscNodes': 0.,
    'meanPerAno': 0., 'eccentricity': 0.,
    'deltaF': delta_F, 'f_min': f_lower, 'f_max': f_max, 'f_ref': fRef,
    'LALparams': params,
    'approximant': ls.NR_hdf5}

    NR_signal = ls.SimInspiralFD(**inspiralFDparams)


    return inspiralFDparams, NR_signal


def GenerateWaveform(m1, m2, S1, S2, mtotal, inclination, approximant, delta_F=0.125, f_lower=20., f_ref=20., params = lal.CreateDict(),distance = 100.*1e6*lal.PC_SI, phiRef = 0.,modes=[]):


    if(len(modes)>0):

        modearray= ls.SimInspiralCreateModeArray()
        for mode in modes:
            ls.SimInspiralModeArrayActivateMode(modearray, mode[0], mode[1])
        ls.SimInspiralWaveformParamsInsertModeArray(params, modearray)


    m1SI = m1 * mtotal / (m1 + m2) * lal.MSUN_SI
    m2SI = m2 * mtotal / (m1 + m2) * lal.MSUN_SI

    fRef=f_ref

    spin1x = S1[0]
    spin1y = S1[1]
    spin1z = S1[2]
    spin2x = S2[0]
    spin2y = S2[1]
    spin2z = S2[2]


    f_max = 2**(int(np.log(3000./delta_F)/np.log(2))+1) * delta_F
    inspiralFDparams = {
    'm1': m1SI, 'm2': m2SI,
    'S1x': spin1x, 'S1y': spin1y, 'S1z': spin1z,
    'S2x': spin2x, 'S2y': spin2y, 'S2z': spin2z,
    'distance': distance, 'inclination': inclination,
    'phiRef': phiRef, 'longAscNodes': 0.,
    'meanPerAno': 0., 'eccentricity': 0.,
    'deltaF': delta_F, 'f_min': f_lower, 'f_max': f_max, 'f_ref': fRef,
    'LALparams': params,
    'approximant': approximant}

    try:
        WF_signal = ls.SimInspiralFD(**inspiralFDparams)
    except RuntimeError:
        raise

    return inspiralFDparams, WF_signal


def GenerateWaveformPars(input_pars,inclination,approximant,LALparams=None):

    wf_params = input_pars.copy()
    wf_params.update({'approximant': approximant})
    wf_params.update({'inclination': inclination})
    if(LALparams!=None):
        wf_params.update({'LALparams': LALparams})

    return wf_params




def RandomSampleMatchPv3(approx1_string, approx2_string, nsample=150., chiMax=0.8, qMin=1., qMax=4,fmin=20., fmax=2048.,
                         mass_bins = np.arange(80,210,40.),final_spin_v=1, distance=100.*1e6*lal.PC_SI, matchdir='',verbose=False,
                         tag='',prec_version=None,seed=None, precession=True, inPlaneOnly=False, precSingleSpin=False, templateTinyPrec=False,
                         phiRefs=np.linspace(0.,2.*np.pi,6), alphas=np.linspace(0.,2.*np.pi,8), zpfact=2, fast_eval=False,
                         mm_threshold=0.95, mode_array_source=None, mode_array_template=None, optimization_method='NM',XPHMconvention=2,twist_phenomhm=False,maxfun=500):


    #run second approximant with default lal parameters
    # make an exception for XPHM, for which we allow to pass the prec_version
    death_counter = 0

    template_lal_pars=lal.CreateDict()
    source_lal_pars=lal.CreateDict()

    if(mode_array_template!=None):
        ModeArray = ls.SimInspiralCreateModeArray()
        for mode in mode_array_template:
            ls.SimInspiralModeArrayActivateMode(ModeArray, mode[0], mode[1])
        ls.SimInspiralWaveformParamsInsertModeArray(template_lal_pars, ModeArray)

    if(mode_array_source!=None):
        ModeArray = ls.SimInspiralCreateModeArray()
        for mode in mode_array_source:
            ls.SimInspiralModeArrayActivateMode(ModeArray, mode[0], mode[1])
        ls.SimInspiralWaveformParamsInsertModeArray(source_lal_pars, ModeArray)

    if(twist_phenomhm == True):
        ls.SimInspiralWaveformParamsInsertPhenomXPHMTwistPhenomHM(template_lal_pars, 1)
        ls.SimInspiralWaveformParamsInsertPhenomXPHMTwistPhenomHM(source_lal_pars, 1)

    f_ref_t=fmin
    f_ref_s=fmin

    f_low_t=fmin
    f_low_s=fmin

    if(approx1_string=='NRSur7dq4'):
        f_ref_s=0.
    if(approx2_string=='NRSur7dq4'):
        f_ref_t=0.
    if(approx1_string=='SEOBNRv4PHM' or approx1_string=='SEOBNRv4P'):
        f_low_s=15.

    #quick hack, need to be refined
    if(prec_version!=None):
        ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(template_lal_pars, prec_version)
        ls.SimInspiralWaveformParamsInsertPhenomXPrecVersion(source_lal_pars,   prec_version)
    if(final_spin_v!=0):
        ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(template_lal_pars, final_spin_v)
        ls.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(source_lal_pars,   final_spin_v)

    if(approx2_string=='IMRPhenomXP' or approx2_string=='IMRPhenomXPHM'):
        ls.SimInspiralWaveformParamsInsertPhenomXPConvention(template_lal_pars, XPHMconvention)
    if(approx1_string=='IMRPhenomXP' or approx1_string=='IMRPhenomXPHM'):
        ls.SimInspiralWaveformParamsInsertPhenomXPConvention(source_lal_pars, XPHMconvention)

    approx1=ls.SimInspiralGetApproximantFromString(approx1_string)
    approx2=ls.SimInspiralGetApproximantFromString(approx2_string)

    sys.stdout.write("# q, s1x, s1y, s1z, s2x, s2y, s2z, inclination, phiRef_s, kappa_s, Mtot, phi_opt, alpha_opt, opt_SNR, mismatch, phi_opt_alt, alpha_opt_alt, time[s], mismatch_alt\n")
    sys.stdout.flush()

    #used in the match optimization
    phases = np.linspace(0., 2.*np.pi, 20, endpoint=False)

    # seed RNG
    np.random.seed(seed=seed)

    for nsim in range(0,nsample):

        if(verbose):
            print('Computing case %d..\n'% nsim)

        q=np.random.uniform(qMin,qMax)
        if(precession):
            chi1mag = np.random.uniform(0, chiMax)

            if(inPlaneOnly):
               cos1th = 0.
            else:
               cos1th =  np.random.uniform(-1.,1.)

            chi1th = np.arccos(cos1th)
            chi1ph = np.random.uniform(0, 2*np.pi)
            chi1 = [chi1mag * np.sin(chi1th) * np.cos(chi1ph), chi1mag * np.sin(chi1th) * np.sin(chi1ph),chi1mag * np.cos(chi1th)]

            if(precSingleSpin):
                 chi2mag = 0.
            else:
                 chi2mag = np.random.uniform(0, chiMax)

            if(inPlaneOnly):
               cos2th = 0.
            else:
               cos2th =  np.random.uniform(-1.,1.)

            chi2th =np.arccos(cos2th)
            chi2ph = np.random.uniform(0, 2*np.pi)
            chi2 = [chi2mag * np.sin(chi2th) * np.cos(chi2ph), chi2mag * np.sin(chi2th) * np.sin(chi2ph), chi2mag * np.cos(chi2th)]
        else:
            chi1mag = np.random.uniform(-chiMax, chiMax)
            chi1 = [0, 0, chi1mag]

            chi2mag = np.random.uniform(-chiMax, chiMax)
            chi2 = [0, 0, chi2mag]


        if(templateTinyPrec):
           chi1[0]=chi1[1]=0
           chi2[0]=chi2[1]=0

        m1=q/(1.+q)
        m2=1./(1.+q)


        inclination = np.arccos(np.random.uniform(-1.,1.))
        phi_ref = np.random.uniform(0, 2*np.pi)
        kappa_s=np.random.uniform(0.,np.pi*0.25)


        addData = False
        for mtotal in mass_bins:

            sys.stdout.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t" % (q, chi1[0],chi1[1],chi1[2],chi2[0],chi2[1],chi2[2],inclination, phi_ref,kappa_s,mtotal))
            sys.stdout.flush()

            try: # if the "signal" is IMRPhenomXP with Pv3 angles, this will catch an error if the angles error
                signal_pars, signal=GenerateWaveform(m1, m2, chi1, chi2, mtotal, inclination, approx1, f_lower=f_low_s,f_ref=f_ref_s, distance=distance,params=source_lal_pars)
            except RuntimeError:
                death_counter += 1
                sys.stdout.write("%f\t%f\t%f\t%f \n" % (-2, -2, -2, -2))
                sys.stdout.flush()
                break


            template_pars=GenerateWaveformPars(signal_pars,inclination,approx2,LALparams=template_lal_pars)
            template_pars['f_min']=f_low_t
            template_pars['f_ref']=f_ref_t

            if(templateTinyPrec):
                  template_pars['S1x']=0.001
                  template_pars['S2y']=0.0005

            start = time.time()
            try:
                #2-step optimization: first a discrete and then a numerical optimization
                if(fast_eval):
                    mismatch = pol_phase_optimized_match(signal, template_pars, ls.SimNoisePSDaLIGOZeroDetHighPower,
                                                     max(fmin,f_low_t,f_low_s), fmax, kappa_s, optimization_method='discrete', rotate_spins=1,test_phi=phiRefs, alphas=alphas, zpfact=zpfact)
                    if(mismatch[2]<mm_threshold):
                        mismatch = pol_phase_optimized_match(signal, template_pars, ls.SimNoisePSDaLIGOZeroDetHighPower,
                                                         max(fmin,f_low_t,f_low_s), fmax, kappa_s, optimization_method=optimization_method, rotate_spins=1,zpfact=zpfact,maxfun=maxfun)
                # numerical optimization straight away
                else:
                        mismatch = pol_phase_optimized_match(signal, template_pars, ls.SimNoisePSDaLIGOZeroDetHighPower,
                                                         max(fmin,f_low_t,f_low_s), fmax, kappa_s, optimization_method=optimization_method, rotate_spins=1,
                                                         zpfact=zpfact, test_phi=phiRefs, alphas=alphas,maxfun=maxfun)

                        if (optimization_method=='SLSQP'):
                             addData=True
                             mismatch1 = pol_phase_optimized_match(signal, template_pars, ls.SimNoisePSDaLIGOZeroDetHighPower,
                                                         max(fmin,f_low_t,f_low_s), fmax, kappa_s, optimization_method='NM', rotate_spins=1,
                                                         zpfact=zpfact, test_phi=phiRefs, alphas=alphas)




            except RuntimeError:
                death_counter += 1
                sys.stdout.write("%f\t%f\t%f\t%f \n" % (-1, -1, -1, -1))
                sys.stdout.flush()
                break


            end = time.time()
            if (addData):
                sys.stdout.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (mismatch[0][0], mismatch[0][1], mismatch[1], mismatch[2], mismatch1[0][0], mismatch1[0][1], end-start, mismatch1[2]))
            else:
                sys.stdout.write("%f\t%f\t%f\t%f 0 0 %f 0\n" % (mismatch[0][0], mismatch[0][1], mismatch[1], mismatch[2], end-start))
            sys.stdout.flush()


# match code to compare precessing model against NR
def NR_SNR_weighted_match(inclinations, nrfile, approx, mass_bins=np.linspace(60,300,5), LALparams=None,
                         fmin=20, fmax=2048, modes=[],
                         kappas=None, phases=None, fMinNR = False, verbose=False,
                         kappa_points = 7, phi_points = 7, delta_F = 0.125, output_file=None, method='dual_a',maxfun=500):
    """
    Code that computes the SNR-weighted phase- and polarization-averaged match between
    an NR signal (provided as an hdf5 file in the lvcnr format) and a template approximant.
    See arXiv:1809.10113 for details!

    Input:
    ======
    inclinations: a list of inclinations at which to evaluate the match code
    nrfile: a string pointing to the location of the NR hdf5 file
    approx: a lalsimulation approximant tag for the template waveform
    Mtotal: the total mass of the system
    LALparams: the lal dictionary used in the generation of the signal/template waveforms
    fmin: lower frequency bound (Hz) for the match code
    fmax: upper frequency bound (Hz) for the match code
    modes: list of modes used in the template/signal generation
    kappas: list of signal polarization angles to be averaged over
    phases: list of signal reference phases to be averaged over
    fMinNR: boolean value to use the starting NR frequency as the lower frequency bound in the match code
            if True, ignores fmin!
    verbose: boolean to turn on/off printing of current location in the match
    kappa_points: if kappas is None, this is the resolution of the polarization sampling used
    phi_points: if phases is None, this is the resolution of the reference phase sampling used
    delta_F: the frequency resolution used

    Output:
    =======
    match_array_inc: array of outputs containing the inclination, SNR-weighted match, and the cumulative match and optSNR data computed at each phase and polarization for this inclination
    """

    if len(kappas)==0:
        kappas = np.linspace(0,np.pi/4,kappa_points)

    if len(phases)==0:
        phases = np.linspace(0,2*np.pi,phi_points)

    #test_phases = np.linspace(0.,2.*np.pi,6)
    #spin_angles = np.linspace(0.,2.*np.pi,8)

    match_array_inc = []

    if(output_file!=None):
        f = open(output_file,"w+")
        f.write('# mtotal, inclination, phase_s, kappa_s, SNR, match\n')

    for Mtotal in mass_bins:
        for inclination in inclinations:
            if verbose:
                print('Running inclination %.6f'%inclination)

            matchData = []
            matchFinal = 0
            SNRFinal = 0
            count=0

            # loop over phases
            for phase_s in phases:
                if verbose:
                    print('\tRunning phase %.6f'%phase_s)

                polMatchData = []
                signal_pars, signal=GenerateNRsignal(nrfile, Mtotal, inclination, phiRef=phase_s, modes=modes, delta_F = delta_F)
                if fMinNR:
                    fmin=signal_pars['f_ref']+5

                template_pars=GenerateWaveformPars(signal_pars,inclination,approx,LALparams=LALparams)

                # loop over polarizations
                for kappa_s in kappas:
                    if verbose:
                        print('\t\tRunning polarization %.6f'%kappa_s)

                    output = pol_phase_optimized_match(signal, template_pars, ls.SimNoisePSDaLIGOZeroDetHighPower,
                                                           max(fmin,signal_pars['f_min']), fmax, kappa_s, optimization_method=method, rotate_spins=1, zpfact=2,maxfun=maxfun)
                    optSNR = output[1]
                    match = output[2]
                    polMatchData.append([kappa_s, optSNR, match])
                    count=count+1
                    matchFinal += (match*optSNR)**3
                    SNRFinal += optSNR**3
                    if(output_file!=None):
                        f.write('%f\t%f\t%f\t%f\t%f\t%f\t'%(Mtotal,inclination,phase_s,kappa_s,optSNR,match))
                        if(count<len(kappas)*len(phases)):
                            f.write('%f\n'% 0.)

                matchData.append([phase_s, polMatchData])
            if(output_file!=None):
                f.write('%f\n'% np.cbrt(matchFinal/SNRFinal))
            match_array_inc.append([inclination, np.cbrt(matchFinal/SNRFinal), matchData])


    if(output_file!=None):
        f.close()

    return match_array_inc
