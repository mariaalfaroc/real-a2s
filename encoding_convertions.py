import os
import re
import numpy as np
import config

""" Kern operations """
class krnConverter():
    def __init__(self):
        self.reserved_words = ['clef', 'k[', '*M']
        self.comment_symbols = ['*', '!']
        

    """ Convert complete kern sequence to CLEAN kern format """
    def cleanKernFile(self, file_path: str) -> list:

        # Reading input file:
        with open(file_path) as fin:
            temp = fin.read().splitlines()

        out_seq = [self.cleanKernToken(u) for u in temp if self.cleanKernToken(u) is not None]

        return out_seq


    """ Convert a kern token to its CLEAN equivalent """
    def cleanKernToken(self, in_token: str) -> str:
        out_token = None # Default

        if any([u in in_token for u in self.reserved_words]): # Relevant reserved tokens
            out_token = in_token

        elif any([in_token.startswith(u) for u in self.comment_symbols]): # Comments
            out_token = None

        elif in_token.startswith('s'): # Slurs
            out_token = 's'

        elif '=' in in_token: # Bar lines
            out_token = '='            

        elif not 'q' in in_token:
            if 'rr' in in_token: # Multirest
                # out_token = re.findall('rr[0-9]+', in_token)[0]
                out_token = None
            elif 'r' in in_token: # Rest
                out_token = in_token.split('r')[0]+'r'
            else: # Music note
                out_token = re.findall('\d+[.]*[a-gA-G]+[n#-]*', in_token)[0]
        return out_token


    """ Convert complete kern sequence to CLEAN and DECOUPLED kern format """
    def cleanAndDecoupleKernFile(self, file_path: str) -> list:

        # Reading input file:
        with open(file_path) as fin:
            temp = fin.read().splitlines()

        out_seq = [self.cleanAndDecoupleKernToken(u) for u in temp]
        out_seq = [x for xs in out_seq for x in xs if x is not None]

        return out_seq


    """ Convert a kern token to its CLEAN and DECOUPLED equivalent """
    def cleanAndDecoupleKernToken(self, in_token: str) -> list:
        out_token = list() # Default

        if any([u in in_token for u in self.reserved_words]): # Relevant reserved tokens
            out_token.append(in_token)

        elif any([in_token.startswith(u) for u in self.comment_symbols]): # Comments
            out_token.append(None)

        elif in_token.startswith('s'): # Slurs
            out_token.append('s')

        elif '=' in in_token: # Bar lines
            out_token.append('=')

        elif not 'q' in in_token:
            if 'rr' in in_token: # Multirest
                # out_token.append(re.findall('rr[0-9]+', in_token)[0])
                out_token.append(None) 
            elif 'r' in in_token: # Rest
                out_token = [in_token.split('r')[0], 'r']
            else: # Notes
                # Extract duration:
                duration = re.findall('\d+', in_token)[0]
                rest = re.split('\d+', in_token)[1]
                out_token.append(duration)

                # Extract dot (if exists):
                dot = [None]
                if '.' in rest:
                    dot = list(re.findall('[.]+', rest)[0])
                    rest = re.split('[.]+', rest)[1]
                out_token.extend(dot)

                # Extract pitch:
                pitch = re.findall('[a-gA-G]+', rest)[0]
                rest = re.split('[a-gA-G]+', rest)[1]
                out_token.append(pitch)

                # Extract alteration (if exists):
                alteration = None
                alteration = re.findall('[n#-]*', rest)[0]
                if len(alteration) == 0: alteration = None 
                out_token.append(alteration)
        else:
            out_token = [None]
        
        return out_token



    """ Convert complete kern sequence to CLEAN and DECOUPLED-with-DOT kern format """
    def cleanAndDecoupleDottedKernFile(self, file_path: str) -> list:

        # Reading input file:
        with open(file_path) as fin:
            temp = fin.read().splitlines()

        out_seq = [self.cleanAndDecoupleDottedKernToken(u) for u in temp]
        out_seq = [x for xs in out_seq for x in xs if x is not None]

        return out_seq


    """ Convert a kern token to its CLEAN and DECOUPLED-with-DOT equivalent """
    def cleanAndDecoupleDottedKernToken(self, in_token: str) -> list:
        out_token = list() # Default

        if any([u in in_token for u in self.reserved_words]): # Relevant reserved tokens
            out_token.append(in_token)

        elif any([in_token.startswith(u) for u in self.comment_symbols]): # Comments
            out_token.append(None)

        elif in_token.startswith('s'): # Slurs
            out_token.append('s')

        elif '=' in in_token: # Bar lines
            out_token.append('=')

        elif not 'q' in in_token:
            if 'rr' in in_token: # Multirest
                # out_token.append(re.findall('rr[0-9]+', in_token)[0])
                out_token.append(None) 
            elif 'r' in in_token: # Rest
                out_token = [in_token.split('r')[0], 'r']
            else: # Notes
                # Extract duration:
                duration = re.findall('\d+', in_token)[0]
                rest = re.split('\d+', in_token)[1]

                # Extract dot (if exists)
                if '.' in rest:
                    dot = list(re.findall('[.]+', rest)[0])
                    duration += "".join(dot)
                    rest = re.split('[.]+', rest)[1]
                out_token.append(duration)

                # Extract pitch:
                pitch = re.findall('[a-gA-G]+', rest)[0]
                rest = re.split('[a-gA-G]+', rest)[1]
                out_token.append(pitch)

                # Extract alteration (if exists):
                alteration = None
                alteration = re.findall('[n#-]*', rest)[0]
                if len(alteration) == 0: alteration = None 
                out_token.append(alteration)
        else:
            out_token = [None]
        
        return out_token



    #### KERN 2 SINGLE LILYPOND ####
    """ Convert complete kern sequence to single lilypond format """
    def cleanLilypondFile(self, file_path: str) -> list:


        # Reading input file:
        with open(file_path) as fin:
            temp = fin.read().splitlines()

        out_seq = [self.cleanLilypondToken(u) for u in temp if self.cleanLilypondToken(u) is not None]

        return out_seq


    """ Convert a particular token from kern to lilypond """
    def cleanLilypondToken(self, in_token: str) -> str:

        out_token = None # Default

        if any([u in in_token for u in self.reserved_words]): # Relevant reserved tokens
            out_token = in_token

        elif any([in_token.startswith(u) for u in self.comment_symbols]): # Comments
            out_token = None

        elif in_token.startswith('s'): # Slurs
            out_token = 's'

        elif '=' in in_token: # Bar lines
            out_token = '='            

        else:
            # Extracting the duration:
            duration = re.findall('[0-9]+', in_token)
            duration = duration[0] if len(duration) == 1 else '4'
            dot = '.' if '.' in in_token else ''

            # Rests:
            if 'rr' in in_token: return # Skipping multirest

            # Regular rests:
            if 'r' in in_token:
                out_token = 'r'
                out_token += duration
                if len(dot): out_token += dot

                return out_token

            # Notes:
            #Obtaining pitch:
            out_token = '<'
            pitch_raw = re.findall('[a-gA-G]+', in_token)[0]
            pitch = "".join(set(pitch_raw.lower()))

            out_token += pitch

            #Obtaining alteration:
            alteration = "".join(re.findall('[n#-]+', in_token))

            if len(alteration): out_token += alteration

            #Obtaining octave:
            if pitch_raw.isupper(): octave = 4 - len(pitch_raw)
            else: octave = len(pitch_raw) + 3
            for _ in range(octave - 2): out_token += "'"
            out_token += '>'

            out_token += duration
            if len(dot): out_token += dot

        return out_token

    #### \KERN 2 SINGLE LILYPOND/ ####

    #### KERN 2 SINGLE LILYPOND ####
    """ Convert complete kern sequence to single lilypond format """
    def cleanLilypondAndDecoupleFile(self, file_path: str) -> str:

        # Reading input file:
        with open(file_path) as fin:
            temp = fin.read().splitlines()

        out_seq = [self.cleanLilypondAndDecoupleToken(u) for u in temp]
        out_seq = [x for xs in out_seq for x in xs if x is not None]

        return out_seq



    """ Convert a particular token from kern to lilypond """
    def cleanLilypondAndDecoupleToken(self, in_token: str) -> list:
        out_token = [None]


        if any([u in in_token for u in self.reserved_words]): # Relevant reserved tokens
            out_token.append(in_token)

        elif any([in_token.startswith(u) for u in self.comment_symbols]): # Comments
            out_token.append(None)

        elif in_token.startswith('s'): # Slurs
            out_token.append('s')

        elif '=' in in_token: # Bar lines
            out_token.append('=')

        else:
            # Extracting the duration:
            duration = re.findall('[0-9]+', in_token)
            duration = duration[0] if len(duration) == 1 else '4'
            dot = '.' if '.' in in_token else ''

            # Rests:
            if 'rr' in in_token: return out_token

            # Regular rests:
            if 'r' in in_token:
                out_token.append('r')
                out_token.append(duration)
                if len(dot): out_token.append(dot)

                return out_token

            # Notes:
            #Obtaining pitch:
            # out_token.append('<')
            pitch_raw = re.findall('[a-gA-G]+', in_token)[0]
            pitch = "".join(set(pitch_raw.lower()))

            out_token.append(pitch)

            #Obtaining alteration:
            alteration = "".join(re.findall('[n#-]+', in_token))

            if len(alteration): out_token.append(alteration)

            #Obtaining octave:
            if pitch_raw.isupper(): octave = 4 - len(pitch_raw)
            else: octave = len(pitch_raw) + 3
            for _ in range(octave - 2): out_token.append("'")
            # out_token.append('>')

            out_token.append(duration)
            if len(dot): out_token.append(dot)

        return out_token

    #### \KERN 2 SINGLE LILYPOND/ ####









    """ Output selector """
    def convert(self, src_file: str , encoding: str) -> list:

        assert encoding in config.encoding_options

        if encoding == 'kern':
            out = self.cleanKernFile(src_file)
        
        elif encoding == 'decoupled':
            out = self.cleanAndDecoupleKernFile(src_file)

        elif encoding == 'decoupled_dot':
            out = self.cleanAndDecoupleDottedKernFile(src_file)

        elif encoding == 'lilypond':
            out = self.cleanLilypondFile(src_file)

        elif encoding == 'lilypond_decoupled':
            out = self.cleanLilypondAndDecoupleFile(src_file)

        return out





if __name__ == '__main__':


    k2l = krnConverter()

    files = [u for u in os.listdir('DATASETS/Sax/krn/') if u.endswith('.krn')]

    for f in files:
        out = k2l.cleanLilypondAndDecoupleFile(os.path.join('DATASETS/Sax/krn/', f))
        print("hello")
