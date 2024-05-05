#%%
import pyttsx3
engine = pyttsx3.init() # object creation

#%%
# Speaking rate
          
engine.setProperty('rate', 200)     # setting up new voice rate
engine.runAndWait()
rate = engine.getProperty('rate')   # getting details of current speaking rate
print (rate)  #printing current voice rate
 
engine.say("Hello World!")
engine.say('My current speaking rate is ' + str(rate))
engine.runAndWait()
#engine.stop()

#%%
engine.setProperty('rate', 125)     # setting up new voice rate
engine.runAndWait()
rate = engine.getProperty('rate')   # getting details of current speaking rate
print (rate)       #printing current voice rate

engine.say("Hello World!")
engine.say('My current speaking rate is ' + str(rate))
engine.runAndWait()
#engine.stop()

#%%
#"""VOLUME"""
engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1
engine.runAndWait()
volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
print (volume)                          #printing current volume level

engine.say("Hello World!")
engine.say('My current speaking volume is ' + str(volume))
engine.runAndWait()
#engine.stop()



#%%
#"""VOLUME"""
engine.setProperty('volume',0.5)    # setting up volume level  between 0 and 1
engine.runAndWait()
volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
print (volume)                          #printing current volume level

engine.say("Hello World!")
engine.say('My current speaking volume is ' + str(volume))
engine.runAndWait()
#engine.stop()


#%%
#"""VOICE"""
voices = engine.getProperty('voices')       #getting details of current voice
engine.setProperty('voice', voices[1].id)  #changing index, changes voices. o for male
engine.runAndWait()
print('My current voice ID is ' + str(voices[1].id))
engine.say("Hello World!")
engine.say('My current voice numeber is 1')
engine.runAndWait()
#engine.stop()

#%%
#"""VOICE"""
voices = engine.getProperty('voices')       #getting details of current voice
engine.setProperty('voice', voices[0].id)   #changing index, changes voices. 1 for female
engine.runAndWait()
print('My current voice ID is ' + str(voices[0].id))

engine.say("Hello World!")
engine.say('My current voice number is 0')
engine.runAndWait()
#engine.stop()


