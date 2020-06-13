# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:27:16 2020

@author: MSI_PC
"""
# Import the required module for text  
# to speech conversion
from gtts import gTTS
text2speech = gTTS(text = 'This is my first text', lang = 'en', slow = False)
text2speech.save(r'C:\Users\MSI_PC\Desktop\Chatbot\Text-to-Speech using gTTS\first_time.mp3')

#The easiest way to get a list of available language is to print them with 
# This module is imported so that we can play the converted audio 
import os 
os.system(r'C:\Users\MSI_PC\Desktop\Chatbot\Text-to-Speech using gTTS\first_time.mp3')

#Returns:
#    dict: A dictionnary of the type `{ '<lang>': '<name>'}`
#
#        Where `<lang>` is an IETF language tag such as `en` or `pt-br`,
#        and `<name>` is the full English name of the language, such as
#        `English` or `Portuguese (Brazil)`
from gtts.lang import tts_langs
supported_languages = tts_langs()

# I'll try in some other language
tts_2 = gTTS(text = 'ma nourriture préférée comme dîner est des pâtes', lang = 'fr-fr', slow = False)
tts_2.save(r'C:\Users\MSI_PC\Desktop\Chatbot\Text-to-Speech using gTTS\third_time.mp3')

tts_en = gTTS('hello', lang='en')
tts_fr = gTTS('bonjour', lang='fr')

with open('hello_bonjour.mp3', 'wb') as f:
       tts_en.write_to_fp(f)
       tts_fr.write_to_fp(f)