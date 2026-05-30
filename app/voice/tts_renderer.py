import asyncio,os,numpy as np
from scipy.signal import butter,sosfilt
SR=44100;TTS_ENGINE=os.getenv("TTS_ENGINE","piper")
async def render_script(text:str,tone:str="calm")->np.ndarray:
    if TTS_ENGINE=="piper":
        proc=await asyncio.create_subprocess_exec("piper","--model",
            os.getenv("PIPER_MODEL","en_US-lessac-medium.onnx"),"--output_raw",
            stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE)
        raw,_=await proc.communicate(input=text.encode())
        audio=np.frombuffer(raw,dtype=np.int16).astype(np.float32)/32768.0
    else:
        import edge_tts,tempfile,soundfile as sf
        vmap={"calm":"en-US-JennyNeural","warm":"en-US-AriaNeural","authoritative":"en-US-GuyNeural","whispering":"en-US-JennyNeural"}
        with tempfile.NamedTemporaryFile(suffix=".mp3",delete=False) as f: path=f.name
        await edge_tts.Communicate(text,vmap.get(tone,"en-US-JennyNeural")).save(path)
        audio,_=sf.read(path,dtype="float32");os.unlink(path)
        if audio.ndim>1: audio=audio[:,0]
    return audio
def apply_eq(audio:np.ndarray,sr:int=SR)->np.ndarray:
    sos=butter(2,6000,'low',fs=sr,output='sos')
    return sosfilt(sos,audio).astype(np.float32)
async def build_meditation_loop(text:str,tone:str,total_secs:int)->np.ndarray:
    audio=await render_script(text,tone); audio=apply_eq(audio)
    rms=np.sqrt(np.mean(audio**2))+1e-9; audio=audio*(10**(-14/20)/rms)
    target=SR*total_secs
    if len(audio)<target: audio=np.concatenate([audio,np.zeros(target-len(audio),np.float32)])
    return audio[:target]
