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












    """ Output selector """
    def convert(self, src_file: str , encoding: str) -> list:

        assert encoding in config.encoding_options

        if encoding == 'kern':
            out = self.cleanKernFile(src_file)
        
        elif encoding == 'decoupled':
            out = self.cleanAndDecoupleKernFile(src_file)

        elif encoding == 'decoupled_dot':
            out = self.cleanAndDecoupleDottedKernFile(src_file)

        return out


if __name__ == '__main__':


    k2l = krnConverter()

    files = [u for u in os.listdir('DATASETS/Sax/krn/') if u.endswith('.krn')]

    for f in files:
        out = k2l.cleanAndDecoupleDottedKernFile(os.path.join('DATASETS/Sax/krn/', f))
        print("hello")
