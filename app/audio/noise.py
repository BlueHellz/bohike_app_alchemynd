import numpy as np
from scipy.signal import lfilter, butter, sosfilt
SR = 44100; CHUNK_SAMP = int(SR * 1.0)

def _white(n): return np.random.randn(n).astype(np.float32)

def _pink(n):
    w = _white(n); b=[0.049922035,-0.095993537,0.050612699,-0.004408786]
    a=[1,-2.494956002,2.017265875,-0.522189400]
    return lfilter(b,a,w).astype(np.float32)

def _brown(n):
    w=_white(n); b=np.cumsum(w); b-=b.mean()
    return (b/(np.abs(b).max()+1e-9)).astype(np.float32)

def _violet(n,sr=SR):
    sos=butter(1,[200],'high',fs=sr,output='sos')
    return sosfilt(sos,_white(n)).astype(np.float32)

def _black(n): return np.zeros(n,np.float32)

def _grey(n,sr=SR):
    sos=butter(2,[200,8000],'band',fs=sr,output='sos')
    return sosfilt(sos,_white(n)).astype(np.float32)

GENERATORS = {"pink":_pink,"brown":_brown,"white":_white,"violet":_violet,"black":_black,"grey":_grey,}

def _norm(x,target_rms):
    rms=np.sqrt(np.mean(x**2))+1e-9; return x*(target_rms/rms)

def generate_noise_chunk(noise_types,blend_weights,n=CHUNK_SAMP,target_rms=0.08):
    out=np.zeros(n,np.float32)
    for nt,w in zip(noise_types,blend_weights):
        raw=GENERATORS.get(nt,_pink)(n)
        out+=_norm(raw,target_rms)*w
    return out
