import re
from typing import List

ENCODING_OPTIONS = ["kern", "decoupled", "decoupled_dot"]

# NOTE
# Multirest are encoded like "rr" in the simplified kern format of the PrIMuS dataset
# but using that encoding would not render (using Verovio) the multi-bar rest as a single bar rest
# If we change the dataset, MAKE SURE OF THIS!
# The SARA dataset does not contain multirests


class krnConverter:
    """Main Kern converter operations class."""

    def __init__(self, encoding: str = "kern") -> None:
        self.reserved_words = ["clef", "k[", "*M"]
        self.comment_symbols = ["*", "!"]

        # Convert function
        assert (
            encoding in ENCODING_OPTIONS
        ), f"You must chose one of the possible encoding options: {','.join(ENCODING_OPTIONS)}"
        self.encoding = encoding
        self.convert_function_options = {
            "kern": self.cleanKernFile,
            "decoupled": self.cleanAndDecoupleKernFile,
            "decoupled_dot": self.cleanAndDecoupleDottedKernFile,
        }
        self.convert_step = self.convert_function_options[self.encoding]

    def cleanKernFile(self, file_path: str) -> List[str]:
        """Convert complete kern sequence to CLEAN kern format."""
        with open(file_path) as fin:
            temp = fin.read().splitlines()
        out_seq = [
            self.cleanKernToken(u) for u in temp if self.cleanKernToken(u) is not None
        ]
        return out_seq

    def cleanKernToken(self, in_token: str) -> str:
        """Convert a kern token to its CLEAN equivalent."""
        out_token = None  # Default

        if any(
            [u in in_token for u in self.reserved_words]
        ):  # Relevant reserved tokens
            out_token = in_token

        elif any([in_token.startswith(u) for u in self.comment_symbols]):  # Comments
            out_token = None

        elif in_token.startswith("s"):  # Slurs
            out_token = "s"

        elif "=" in in_token:  # Bar lines
            out_token = "="

        elif not "q" in in_token:
            if "rr" in in_token:  # Multirest
                out_token = re.findall("rr[0-9]+", in_token)[0]
            elif "r" in in_token:  # Rest
                out_token = in_token.split("r")[0] + "r"
            else:  # Music note
                out_token = re.findall("\d+[.]*[a-gA-G]+[n#-]*", in_token)[0]

        return out_token

    # ---------------------------------------------------------------------------- DECOUPLE

    def cleanAndDecoupleKernFile(self, file_path: str) -> List[str]:
        """Convert complete kern sequence to CLEAN and DECOUPLED kern format."""
        with open(file_path) as fin:
            temp = fin.read().splitlines()
        out_seq = [self.cleanAndDecoupleKernToken(u) for u in temp]
        out_seq = [x for xs in out_seq for x in xs if x is not None]
        return out_seq

    def cleanAndDecoupleKernToken(self, in_token: str) -> List[str]:
        """Convert a kern token to its CLEAN and DECOUPLED equivalent."""
        out_token = []  # Default

        if any(
            [u in in_token for u in self.reserved_words]
        ):  # Relevant reserved tokens
            out_token.append(in_token)

        elif any([in_token.startswith(u) for u in self.comment_symbols]):  # Comments
            out_token.append(None)

        elif in_token.startswith("s"):  # Slurs
            out_token.append("s")

        elif "=" in in_token:  # Bar lines
            out_token.append("=")

        elif not "q" in in_token:
            if "rr" in in_token:  # Multirest
                out_token.append(re.findall("rr[0-9]+", in_token)[0])
            elif "r" in in_token:  # Rest
                out_token = [in_token.split("r")[0], "r"]
            else:  # Notes
                # Extract duration:
                duration = re.findall("\d+", in_token)[0]
                rest = re.split("\d+", in_token)[1]
                out_token.append(duration)

                # Extract dot (if exists):
                dot = [None]
                if "." in rest:
                    dot = list(re.findall("[.]+", rest)[0])
                    rest = re.split("[.]+", rest)[1]
                out_token.extend(dot)

                # Extract pitch:
                pitch = re.findall("[a-gA-G]+", rest)[0]
                rest = re.split("[a-gA-G]+", rest)[1]
                out_token.append(pitch)

                # Extract alteration (if exists):
                alteration = None
                alteration = re.findall("[n#-]*", rest)[0]
                if len(alteration) == 0:
                    alteration = None
                out_token.append(alteration)
        else:
            out_token = [None]

        return out_token

    # ---------------------------------------------------------------------------- DECOUPLE DOTTED

    def cleanAndDecoupleDottedKernFile(self, file_path: str) -> List[str]:
        """Convert complete kern sequence to CLEAN and DECOUPLED-with-DOT kern format."""
        with open(file_path) as fin:
            temp = fin.read().splitlines()
        out_seq = [self.cleanAndDecoupleDottedKernToken(u) for u in temp]
        out_seq = [x for xs in out_seq for x in xs if x is not None]
        return out_seq

    def cleanAndDecoupleDottedKernToken(self, in_token: str) -> List[str]:
        """Convert a kern token to its CLEAN and DECOUPLED-with-DOT equivalent."""
        out_token = []  # Default

        if any(
            [u in in_token for u in self.reserved_words]
        ):  # Relevant reserved tokens
            out_token.append(in_token)

        elif any([in_token.startswith(u) for u in self.comment_symbols]):  # Comments
            out_token.append(None)

        elif in_token.startswith("s"):  # Slurs
            out_token.append("s")

        elif "=" in in_token:  # Bar lines
            out_token.append("=")

        elif not "q" in in_token:
            if "rr" in in_token:  # Multirest
                out_token.append(re.findall("rr[0-9]+", in_token)[0])
            elif "r" in in_token:  # Rest
                out_token = [in_token.split("r")[0], "r"]
            else:  # Notes
                # Extract duration:
                duration = re.findall("\d+", in_token)[0]
                rest = re.split("\d+", in_token)[1]

                # Extract dot (if exists)
                if "." in rest:
                    dot = list(re.findall("[.]+", rest)[0])
                    duration += "".join(dot)
                    rest = re.split("[.]+", rest)[1]
                out_token.append(duration)

                # Extract pitch:
                pitch = re.findall("[a-gA-G]+", rest)[0]
                rest = re.split("[a-gA-G]+", rest)[1]
                out_token.append(pitch)

                # Extract alteration (if exists):
                alteration = None
                alteration = re.findall("[n#-]*", rest)[0]
                if len(alteration) == 0:
                    alteration = None
                out_token.append(alteration)
        else:
            out_token = [None]

        return out_token

    # ---------------------------------------------------------------------------- CONVERT CALL

    def convert(self, src_file: str) -> List[str]:
        out = self.convert_step(src_file)
        return out


####################################################################################################


def decoupledDotKern2Kern(in_seq: List[str]) -> List[str]:
    out_seq = []

    it = 0
    while it < len(in_seq):
        if in_seq[it].startswith("*") or in_seq[it] == "=":
            out_seq.append(in_seq[it])
            it += 1
        else:
            new_token = ""

            # Duration:
            extract_duration = False
            while not extract_duration and it < len(in_seq):
                try:
                    int(in_seq[it].split(".")[0])
                    new_token = in_seq[it]
                    extract_duration = True
                except:
                    pass
                it += 1

            # Pitch:
            extract_pitch = False
            while not extract_pitch and it < len(in_seq):
                if list(set(in_seq[it].lower()))[0] in [
                    "a",
                    "b",
                    "c",
                    "d",
                    "e",
                    "f",
                    "g",
                    "r",
                ]:
                    new_token += in_seq[it]
                    extract_pitch = True
                it += 1

            # Alteration:
            if it < len(in_seq):
                if "-" in in_seq[it] or "#" in in_seq[it]:
                    new_token += in_seq[it]
                    it += 1

                out_seq.append(new_token)

    return out_seq


def decoupledKern2Kern(in_seq: List[str]) -> List[str]:
    out_seq = []

    it = 0
    while it < len(in_seq):
        if in_seq[it].startswith("*") or in_seq[it] == "=":
            out_seq.append(in_seq[it])
            it += 1
        else:
            new_token = ""

            # Duration:
            extract_duration = False
            while not extract_duration and it < len(in_seq):
                try:
                    int(in_seq[it])
                    new_token = in_seq[it]
                    extract_duration = True
                except:
                    pass
                it += 1

            extract_dot = False
            while not extract_dot and it < len(in_seq):
                if in_seq[it] == ".":
                    new_token += "."
                    it += 1
                elif list(set(in_seq[it].lower()))[0] in [
                    "a",
                    "b",
                    "c",
                    "d",
                    "e",
                    "f",
                    "g",
                    "r",
                ]:
                    extract_dot = True
                else:
                    it += 1

            # Pitch:
            extract_pitch = False
            while not extract_pitch and it < len(in_seq):
                if list(set(in_seq[it].lower()))[0] in [
                    "a",
                    "b",
                    "c",
                    "d",
                    "e",
                    "f",
                    "g",
                    "r",
                ]:
                    new_token += in_seq[it]
                    extract_pitch = True
                it += 1

            # Alteration:
            if it < len(in_seq):
                if "-" in in_seq[it] or "#" in in_seq[it]:
                    new_token += in_seq[it]
                    it += 1

                out_seq.append(new_token)

    return out_seq
