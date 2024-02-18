#!/usr/bin/env python
# coding: utf-8

# # ドレミファソラシドを鳴らす
# 音を鳴らすためには次のパッケージを用いる。   
# PyAudio https://people.csail.mit.edu/hubert/pyaudio/ 
# 
# 備考：国によって「音名」と「階名」の呼び方は異なる。  
# 日本人が良く知っているのは音名であり，これはイタリア語が由来である。<br>
# イタリア語の呼び方は，ドレミファソラシド（Do, Re, Mi, Fa, Sol, La, Si）<br>
# ドイツ語では　C（ツェー），D（デー），E（エー），F（エフ），G（ゲー），A（アー），H（ハー）<br>
# 英語では，C,D,E,F,G,A,Bである。<br>
# Wikipedia https://en.wikipedia.org/wiki/Key_signature_names_and_translations <br>
# YAMAHA コードについて学ぶ https://jp.yamaha.com/services/music_pal/study/chord/index.html<br>
# YAMAHA 楽譜について学ぶ  https://jp.yamaha.com/services/music_pal/study/score/index.html

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
import pyaudio


# #### 十二平均律　 twelve-tone equal temperament
# http://stby.jp/heikinritu.html<br>
# https://en.wikipedia.org/wiki/Equal_temperament

# In[ ]:


base = 440 # A
r12 = 2**(1/12) # rate of twelve-tone equal temperament
# Pitch Names: C, D, E, F, G, A(base), B, C (English Form)
Pname = [base/(r12**9), base/(r12**7), base/(r12**5), base/(r12**4), base/(r12**2), \
                base, base*(r12**2), base*(r12**3)]
Pname


# In[ ]:


SRATE = 44100  # Sampling rate
BPM  = 100    # Beats Per Minute
N4   = 60/BPM # 4分音符, Quarter note
N1, N2, N8 = 4*N4, 2*N4, N4/2
print(N1, N2, N4, N8)


# In[ ]:


# Make a sine wave
def makeSineWave(freq, length, amp):
#    slen = int(length * SRATE)
#    t = float(freq) * np.pi * 2 / SRATE
#    return np.sin(np.arange(slen) * t) * amp
    nsample = np.arange(int(length*SRATE)) # the number of sampling
    return amp*np.sin((2*np.pi*float(freq)/SRATE)*nsample)


# Play
def play_wave(stream, samples):
    stream.write(samples.astype(np.float32).tostring())
    
# PyAudioのストリームを開く
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=SRATE,
                frames_per_buffer=1024, output=True)


# #### 1音ずつ鳴らす

# In[ ]:


play_wave(stream, makeSineWave(Pname[0], N4, 1.0))
play_wave(stream, makeSineWave(Pname[2], N4, 1.0))
play_wave(stream, makeSineWave(Pname[1], N4, 1.0))
play_wave(stream, makeSineWave(Pname[4], N4*3, 1.0))
play_wave(stream, makeSineWave(Pname[4], N4, 1.0))
play_wave(stream, makeSineWave(Pname[1], N4, 1.0))
play_wave(stream, makeSineWave(Pname[2], N4, 1.0))
play_wave(stream, makeSineWave(Pname[0], N4*3, 1.0))


# In[ ]:





# In[ ]:





# #### 和音（ドミソ）を鳴らす

# In[ ]:


def makeChord(t):
    nsample = np.arange(int(t*SRATE))
#    chd = np.zeros(int(t*SRATE))
    c = np.sin( (2*np.pi*Pname[0]/SRATE)*nsample)
    e = np.sin( (2*np.pi*Pname[2]/SRATE)*nsample)
    g = np.sin( (2*np.pi*Pname[4]/SRATE)*nsample)
    chd = (c+e+g)/np.max(c+e+g) # Normalization
    return chd


# In[ ]:


play_wave(stream, makeChord(2))


# In[ ]:



