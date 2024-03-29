from librosa import load
from moviepy.editor import AudioFileClip
from os import path
from scipy.io.wavfile import write
import dawdreamer as daw

VALID_FILETYPES = ['.mp3', '.wav'] #These are the only filetypes that I know work for sure
SAMPLE_RATE = 44100
BUFFER_SIZE = 512

def getConjoiner():
    currentDir = path.dirname(__file__)
    if '\\' in currentDir:
        conjoiner = '\\'
    elif '/' in currentDir:
        conjoiner = '/'
    return conjoiner

def loadAudioFile(file_path, duration=None):
    '''
    Loads a .wav or .mp3 file and translates it into data that dawdreamer (the digital audio workspace we're using) can understand
    '''
    sig, rate = load(file_path, duration=duration, mono=False, sr=SAMPLE_RATE)
    assert(rate == SAMPLE_RATE)
    return sig

def loadPreset(file, presetOption):
    '''
    Loads a preset from a textfile.

    Textfile syntax should be as follows:
    
        FIRST LINE:
            Name: Pitch Shift
            Value: False or any integer from -12 to 12
        
        SECOND LINE:
            Name: Bass Boost
            Value: False or list of four floats from 0.0 to 1.0
        
        THIRD LINE:
            Name: Reverb
            Value: False or list of four floats from 0.0 to 1.0
        
        FOURTH LINE:
            Name: Soft Clipper
            Value: False or list of five floats from 0.0 to 1.0
        
        EXAMPLE PRESET AS TEXTFILE:

            False
            0.0 1.0 1.0 1.0
            False
            0.1 0.1 0.1 0.1 0.2
    '''
    conjoiner = getConjoiner()
    presetInput = open(path.dirname(__file__)+conjoiner+'Presets'+conjoiner+presetOption+'.txt').readlines()

    for counter in range(len(presetInput)):
        presetInput[counter] = presetInput[counter].strip()

        if presetInput[counter] == 'False':
            presetInput[counter] = False
        else:
            presetInput[counter] = presetInput[counter].split(' ')

            for number in range(len(presetInput[counter])):
                presetInput[counter][number] = float(presetInput[counter][number])

    megafyFile(file, presetInput[0], presetInput[1], presetInput[2], presetInput[3])

def megafyFile(file, PITCH_SHIFT_CHOICE=False, BASS_BOOST_CHOICE=False, REVERB_CHOICE=False, SOFT_CLIPPER_CHOICE=False):
    '''
        DESCRIPTION:

            Megafies a given file using inputted parameters.

            NOTE (FOR EVERYTHING EXCEPT PITCH_SHIFT_CHOICE): All parameters work with values from 0.0 to 1.0. This means 0.5 could represent something very different depending on the context.
            NOTE (FOR PITCH_SHIFT_CHOICE): Audio shouldn't be shifted above +12 semitones and less than -12 semitones.

        RETURNS:

            Returns True if everything works.

        PARAMETERS:

            file : str
                Absolute path of the file (.wav or .mp3) that will be Megafied
            
            PITCH_SHIFT_CHOICE : int

                PITCH_SHIFT_CHOICE is False by default.
                PITCH_SHIFT_CHOICE should be an int representing the number of semitones they'd like to shift their audio.
                
                Examples:

                    If user would like to raise audio by 5 semitones, PITCH_SHIFT_CHOICE = 5
                    If user would like to lower audio by 7 semitones, PITCH_SHIFT_CHOICE = -7
            
            BASS_BOOST_CHOICE : list[float, float, float, float]

                BASS_BOOST_CHOICE is False by default.
                BASS_BOOST_CHOICE should be an list of four floats, each representing a different value:
            
                    BASS_BOOST_CHOICE[0]:
                        Name: Output Gain
                        Description: When output gain is raised above its default value, the audio is made artifically louder. When lowered, the audio is quieter.
                        Unit: dB (Decibels)
                        Default Value: 0.5 (+/- 0 dB, aka no change to output gain)
                        Possible Value Range: 0.0 to 1.0 (0.0 means -20 dB and 1.0 means +20 dB)
                        Recommended Value: For a good megafy effect, I'd suggest 0.5 to 0.525 (0 to +1 dB).
                
                    BASS_BOOST_CHOICE[1]:
                        Name: Frequency
                        Description: Which frequency the bass boost will target
                        Unit: Hz (Hertz)
                        Default Value: 0.327654 (80 hZ)
                        Possible Value Range: 0.0 to 1.0 (0.0 means 10 Hz and 1.0 means 2000 Hz)
                        Recommended Value: For a good megafy effect, I'd suggest ~0.272 (~60 Hz)
                        NOTE: The Hz scale is not linear.
                    
                    BASS_BOOST_CHOICE[2]:
                        Name: Boost
                        Description: The actual bass boost.
                        Unit: dB (Decibel)
                        Default Value: 0 (No boost)
                        Possible Value Range: 0.0 to 1.0 (0.0 means 0.0 dB and 1.0 means +18.0 dB)
                        Recommended Value: For a good megafy effect, I'd suggest ~0.7 (+12.6 dB)
                        
                    BASS_BOOST_CHOICE[3]:
                        Name: Mode
                        Description: The type of bass boost
                        Unit: N/A
                        Default Value: 0.0
                        Possible Value Ranges: 0.0 (Classic), 0.5 (Passive) and 1.0 (Combo)
                        Recommended Value: For a good megafy effect, I'd suggest choosing 0.5 (Passive)
                        NOTE: BASS_BOOST_CHOICE[3] cannot be anything but either 0.0, 0.5, or 1.0 
            
            REVERB_CHOICE : list[float, float, float, float]

                REVERB_CHOICE is False by default.
                REVERB_CHOICE should be an list of four floats, each representing a different value:
                    
                    REVERB_CHOICE[0]:
                        Name: Reverb
                        Description: Amount of reverb
                        Unit: % (Percent)
                        Default Value: 1.0 (Very wet)
                        Possible Value Ranges: 0.0 to 1.0 (0.0 means no change and 1.0 means a lot of change)
                        Recommended Value: For a good megafy effct, I'd suggest no reverb at all (0.0)
                    
                    REVERB_CHOICE[1]:
                        Name: Wide
                        Description: Wideness of audio
                        Unit: % (Percent)
                        Default Value: 0.333333 (No change)
                        Possible Value Ranges: 0.0 to 1.0 (0.0 means completely narrow (mono) and 1.0 means 200% (very stereo))
                        Recommended Value: For a good megafy effect, I'd suggest no widening at all (0.333333)
                    
                    REVERB_CHOICE[2]:
                        Name: High-Pass
                        Description: How much of the high end to cut off
                        Unit: Hz (Hertz)
                        Default Value: 0.0
                        Possible Value Ranges: 0.0 to 1.0 (0.0 means no cut off and 1.0 means cut off at 20000 Hz)
                        Recommended Value: For a good megafy effect, I'd suggest no high-pass at all (0.0)
                    
                    REVERB_CHOICE[3]:
                        Name: Low-Pass
                        Description: How much of the low end to cut off
                        Unit: Hz (Hertz)
                        Default Value: 1.0
                        Possible Value Ranges: 0.0 to 1.0 (0.0 means cut off at 20 Hz and 1.0 means no cut off)
                        Recommended Value: For a good megafy effect, I'd suggest no low-pass at all (1.0)
                    
            SOFT_CLIPPER_CHOICE : list[float, float, float, float, float]

                SOFT_CLIPPER_CHOICE is False by default.
                SOFT_CLIPPER_CHOICE should be an list of five floats, each representing a different value:

                    SOFT_CLIPPER_CHOICE[0]:
                        Name: Threshold
                        Description: The level where the soft clipper will intervene
                        Unit: dB (Decibel)
                        Default Value: 1.0 (Threshold at ceiling aka +0.00 dB)
                        Possible Value Ranges: 0.0 to 1.0 (0.0 being -10.00 dB and 1.0 being +0.00 dB)
                        Recommended Value: For a good megafy effect, I'd suggest no change to threshold (1.0)
                    
                    SOFT_CLIPPER_CHOICE[1]:
                        Name: Input Gain
                        Description: Louden audio before soft clipping initiates
                        Unit: dB (Decibel)
                        Default Value: 0.5
                        Possible Value Ranges: 0.0 to 1.0 (0.0 being -9.0 dB and 1.0 being +9.0 dB)
                        Recommended Value: For a good megafy effect, I'd suggest no change to input gain (0.5)

                    SOFT_CLIPPER_CHOICE[2]:
                        Name: Positive Saturation
                        Description: Not really sure lol. Fun to experiment with though.
                        Unit: N/A
                        Default Value: 0.0
                        Possible Value Ranges: 0.0 to 1.0 (0.0 means 0.1 and 1.0 means 2.0)
                        Recommended Value: For a good megafy effect, I'd suggest leaving positive saturation alone (0.0)
                    
                    SOFT_CLIPPER_CHOICE[3]:
                        Name: Negative Saturation
                        Description: Not really sure lol. Fun to experiment with though.
                        Unit: N/A
                        Default Value: 0.0
                        Possible Value Ranges: 0.0 to 1.0 (0.0 means 0.1 and 1.0 means 2.0)
                        Recommended Value: For a good megafy effect, I'd suggest leaving negative saturation alone (0.0)
                    
                    SOFT_CLIPPER_CHOICE[4]:
                        Name: Saturation
                        Description: Makes the sound fatter
                        Unit: Boolean (True/False)
                        Default Value: 1.0
                        Possible Value Ranges: 0.0 (False) or 1.0 (True)
                        Recommended Value: For a good megafy effect, I'd suggest 1.0 (True)

    '''
    conjoiner = getConjoiner()

    #What's running everything
    engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)

    #Graph is the order in which we add different effects. Reverbed is just a status to see if the audio's been reverbed yet or not
    OUR_GRAPH = []
    reverbed = False
    lowered = False

    #Turn song into understandable language
    song = loadAudioFile(file)

    #Set pitch shift and its parameters
    if PITCH_SHIFT_CHOICE != False:
        lowered = True
        PITCH_SHIFT_CHOICE = PITCH_SHIFT_CHOICE[0]
        tranposeValue = PITCH_SHIFT_CHOICE
        playback_processor = engine.make_playbackwarp_processor("my_playback", song)
        playback_processor.transpose = tranposeValue
        playback_processor.time_ratio = 2**(-tranposeValue/12)

        playback_processor.set_options(
            daw.PlaybackWarpProcessor.option.OptionTransientsSmooth |
            daw.PlaybackWarpProcessor.option.OptionPitchHighQuality |
            daw.PlaybackWarpProcessor.option.OptionChannelsTogether
        )

        OUR_GRAPH.append((playback_processor, []))
    else:
        playback_processor = engine.make_playback_processor("my_playback", song)
        OUR_GRAPH.append((playback_processor, []))

    #Set bass booster and its parameters
    if BASS_BOOST_CHOICE != False:
        bass_boost = engine.make_plugin_processor("my_bass_boost", path.dirname(__file__)+conjoiner+'Plugins'+conjoiner+'BarkOfDog2.dll')
        bass_boost.set_parameter(2, BASS_BOOST_CHOICE[0]) #Output gain (dB)
        bass_boost.set_parameter(3, BASS_BOOST_CHOICE[1]) #Freq. (Hz)
        bass_boost.set_parameter(4, BASS_BOOST_CHOICE[2]) #Boost (dB)
        bass_boost.set_parameter(5, BASS_BOOST_CHOICE[3]) #Mode (0.0==Classic, 0.5==Passive, 1.0==Combo)
        OUR_GRAPH.append((bass_boost, ["my_playback"]))

    #Set reverb levels and its parameters
    if REVERB_CHOICE != False:           
        reverb = engine.make_plugin_processor("my_reverb", path.dirname(__file__)+conjoiner+'Plugins'+conjoiner+'MConvolutionEZ.dll')
        reverb.set_parameter(0, REVERB_CHOICE[0]) #Reverb (0==Super Dry, 1==Super Wet)
        reverb.set_parameter(1, REVERB_CHOICE[1]) #Wide (0.333333==Off, 0==Mono, 1==200%)
        reverb.set_parameter(2, REVERB_CHOICE[2]) #High-pass (0==Off)
        reverb.set_parameter(3, REVERB_CHOICE[3]) #Low-pass (1==Off)

        reverbed = True

        if len(OUR_GRAPH) == 2:
            OUR_GRAPH.append((reverb, ["my_bass_boost"]))
        elif len(OUR_GRAPH) == 1:
            OUR_GRAPH.append((reverb, ["my_playback"]))

    #Set soft clipper and its parameters
    if SOFT_CLIPPER_CHOICE != False:
        soft_clipper = engine.make_plugin_processor("my_soft_clipper", path.dirname(__file__)+conjoiner+'Plugins'+conjoiner+'Initial Clipper.dll')
        soft_clipper.set_parameter(0, SOFT_CLIPPER_CHOICE[0]) #Threshold
        soft_clipper.set_parameter(1, SOFT_CLIPPER_CHOICE[1]) #Input gain
        soft_clipper.set_parameter(2, SOFT_CLIPPER_CHOICE[2]) #Positive saturation
        soft_clipper.set_parameter(3, SOFT_CLIPPER_CHOICE[3]) #Negative saturation
        soft_clipper.set_parameter(4, SOFT_CLIPPER_CHOICE[4]) #Saturate (0.0==False, 1.0==True)

        if len(OUR_GRAPH) == 3:
            OUR_GRAPH.append((soft_clipper, ["my_reverb"]))
        elif len(OUR_GRAPH) == 2 and reverbed == True:
            OUR_GRAPH.append((soft_clipper, ["my_reverb"]))
        elif len(OUR_GRAPH) == 2 and reverbed != True:
            OUR_GRAPH.append((soft_clipper, ["my_bass_boost"]))
        elif len(OUR_GRAPH) == 1:
            OUR_GRAPH.append((soft_clipper, ["my_playback"]))

    #Load graph onto engine
    engine.load_graph(OUR_GRAPH)

    #Render clip
    durationOfClip = AudioFileClip(file)
    if lowered == True:
        durationOfClip = (durationOfClip.duration)*(2**(-tranposeValue/12))
    else:
        durationOfClip = durationOfClip.duration
    engine.render(durationOfClip)
        
    #Extract audio from engine
    audio = engine.get_audio()

    write(path.dirname(__file__)+conjoiner+'Output'+conjoiner+str(file[file.rfind(conjoiner)+1:]).upper()[:-4]+'.wav', SAMPLE_RATE, audio.transpose())

    return True 

# TESTING TESTING TESTING
loadPreset(r'C:\Users\samlb\Documents\DOWNLOAD_YOUTUBE\Output\Star Shopping.mp3', 'Default - Copy (2)')
# megafyFile(r'C:\Users\samlb\Documents\DOWNLOAD_YOUTUBE\Output\Glaive - 1984 (Directed by Cole Bennett).mp3', PITCH_SHIFT_CHOICE=[3], BASS_BOOST_CHOICE=[0.75,0.272, 0.8, 0.5], SOFT_CLIPPER_CHOICE=[1.0, 0.5, 0.0, 0.0, 1.0])