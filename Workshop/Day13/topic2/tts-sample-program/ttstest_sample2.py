#%%
#Speaking text

import pyttsx3
engine = pyttsx3.init()
engine.say('Sally sells seashells by the seashore.')
engine.say('The quick brown fox jumped over the lazy dog.')
engine.runAndWait()


#%%
#Listening for events

import pyttsx3
def onStart(name):
   print ('starting', name)
def onWord(name, location, length):
   #print ('word', name, location, length)
   print ('word', name, message[location:location+length+1])
def onEnd(name, completed):
   print ('finishing', name, completed)
   
message = 'The quick brown fox jumped over the lazy dog.'
  
engine = pyttsx3.init()
dict1 = engine.connect('started-utterance', onStart)
dict2 = engine.connect('started-word', onWord)
dict3 = engine.connect('finished-utterance', onEnd)
engine.say(message)
engine.runAndWait()
engine.disconnect(dict1)
engine.disconnect(dict2)
engine.disconnect(dict3)



#%%
#Interrupting an utterance

import pyttsx3
def onWord(name, location, length):
   print ('word', name, location, length)
   if location > 10:
      engine.stop()
engine = pyttsx3.init()
dict1 = engine.connect('started-word', onWord)
engine.say('The quick brown fox jumped over the lazy dog.')
engine.runAndWait()
engine.disconnect(dict1)

#%%

#Running a driver event loop
import pyttsx3

def onStart(name): 
   print ('starting', name)
def onWord(name, location, length):
   print ('word', name, location, length)
def onEnd(name, completed):
   print ('finishing', name, completed)
   if name == 'fox':
      engine.say('What a lazy dog!', 'dog')
   elif name == 'dog':
      engine.endLoop()
      
engine = pyttsx3.init()
dict1 = engine.connect('started-utterance', onStart)
dict2 = engine.connect('started-word', onWord)
dict3 = engine.connect('finished-utterance', onEnd)
engine.say('The quick brown fox jumped over the lazy dog.', 'fox')
engine.startLoop()
engine.runAndWait()
#engine.endLoop()
#engine.stop()
engine.disconnect(dict1)
engine.disconnect(dict2)
engine.disconnect(dict3)


