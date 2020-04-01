# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:53:08 2020

@author: MSI_PC
"""

import speech_recognition as sr
stt = sr.Recognizer()

# We'll be using Google Cloud Speech API for the conversion part

# To convert a saved audio file to text (supported- .wav, .flac format)
import pydub
from pydub import AudioSegment # Convert mp3 to wav 
import os   # to read mp3 files


converted = "output.wav"
# convert wav to mp3                                                            
sound = AudioSegment.from_mp3("D:/Music/Some hindi song.mp3")
sound.export(converted, format="wav")

with sr.AudioFile("output.wav") as source:
    # listen for the data (load audio to memory)
    stt.adjust_for_ambient_noise(source)
    audio_data = stt.record(source, duration= 10, offset = 46)
    # recognize (convert from speech to text)
    
    try:
           text = stt.recognize_google(audio_data, language = 'hi-IN')       #Since song was hindi song
    # https://cloud.google.com/speech-to-text/docs/languages [set of suported languages]  
    #error occurs when google could not understand what was said 
    except : 
           print("Google Speech Recognition could not understand audio") 
           
# Using the MicroPhone
# Pyaudio is used for this
import pyaudio
mic=sr.Microphone()
with mic as source1:
       stt.adjust_for_ambient_noise(source1)
       input_from_mic = stt.listen(source1, phrase_time_limit= 45, timeout= 4)

try:
       text = stt.recognize_google(input_from_mic, language = 'hi-IN')
except:
       print("Google Speech Recognition could not understand audio")
              
              
       


