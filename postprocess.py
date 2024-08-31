import json
import re
import argparse
from typing import List, Tuple

# argument parser
parser = argparse.ArgumentParser(prog="postprocess", description="Prostprocess the data.")

parser.add_argument("--path", type=str,help="data file path")
parser.add_argument("--ensemble_path_1", type=str, default=None, help="ensemble1 data file path")
parser.add_argument("--ensemble_path_2", type=str, default=None, help="ensemble2 data file path")
parser.add_argument("--output_path", type=str,help="output file path")

"""
Example:

python postprocess_galaxy.py --path ./data.json --ensemble_path_1 ./data2.json --output_path ./data_postprocessed.json
"""

# Postprocess the data
def postprocess(data:json) -> json:
    """
    1. Remove '## 전반적인 요약', '## speaker_2 요약', '## speaker_2 요약' from the data
    2. Concatenate the summaries into one string
    """
    for example in data:
        output = example["output"]
        speakers = set()
        for cvt in example["input"]["conversation"]:
            speakers.add(cvt["speaker"])
        speaker_1, speaker_2 = speakers

        output = re.sub(r'## 전반적인 요약', '', output)
        output = re.sub(r'## ' + speaker_1 + ' 요약', '', output)
        output = re.sub(r'## ' + speaker_2 + ' 요약', '', output)
        output = re.sub(r'\s+', ' ', output)
        output = output.strip()

        example["output"] = output

    return data


# Ensemble two json data and select the shorter one
def ensemble_1(data1:json, data2_path:str) -> json:
    """
    ensemble two json data
    Compare data1 and data2 and select the shorter one
    """
    # Load data1
    with open(data2_path, 'r') as f:
        data2 = json.load(f)

    # Compare data1 and data2 and select the shorter one
    for i in range(len(data1)):
        if len(data1[i]["output"]) > len(data2[i]["output"]):
            data1[i]["output"] = data2[i]["output"]
    
    return data1


# find the indexes to split the structured summary (total_summary, speaker_1_summary, speaker_2_summary)
def find_split_indexes(text: str) -> List[Tuple[int, int]]:
        """
        Find the indexes(strat, end) to split the structured summary.
        """
        # The number of 'SD{7}[은는]{1}'
        num_speakers = len(re.findall(r'SD\d{7}[은는]{1}', text))

        # Split the structured summary based on the number of 'SD{7}[은는]{1}'
        if num_speakers == 2: 
            mathes = re.finditer(r'SD\d{7}[은는]{1}', text)
            return [(match.group(), match.start()) for match in mathes] # [(speaker1, start_id_1), (speaker2, start_id_2)]
        
        elif num_speakers in [0, 1]:
            matches = re.finditer(r'SD\d{7}\w+', text)

            first_match = next(matches)
            first_tuple = (first_match.start(), first_match.group())

            for match in matches:
                if match.group()[:9] == first_tuple[1][:9]: # SD{7}가 같은 경우
                    continue
                return [(first_tuple[1], first_tuple[0]), (match.group(), match.start())]
            
        elif num_speakers >= 3:
            matches = re.finditer(r'SD\d{7}[은는]{1}', text)

            first_match = next(matches)
            first_tuple = (first_match.start(), first_match.group())

            for match in matches:
                if match.group()[:9] == first_tuple[1][:9]: # SD{7}가 같은 경우
                    continue
                return [(first_tuple[1], first_tuple[0]), (match.group(), match.start())]


# find the index to split the structured summary (total_summary, speaker_2_summary)
def find_speaker_2_start_index(text: str) -> Tuple[str, int]:
    """
    Find the indexes(strat, end) to split the structured summary.
    """
    # Find the speaker_2
    matches = re.finditer(r'SD\d{7}\w+', text)

    first_match = next(matches)
    first_tuple = (first_match.start(), first_match.group())

    return (first_tuple[1], first_tuple[0]) # (speaker_2, start_index)


def ensemble_2(data1:json, data2_path:str) -> json:
    """
    ensemble two json data
    Compare data1's speaker 2 summary and data2's speaker 2 summary and select the shorter one
    """
    # Load data1
    with open(data2_path, 'r') as f:
        data2 = json.load(f)

    # Compare data1's speaker 2 summary and data2's speaker 2 summary and select the shorter one
    for i in range(len(data1)):
        text1 = data1[i]["output"]
        text2 = data2[i]["output"]

        # Find the indexes to split the structured summary
        split_indexes_1 = find_split_indexes(text1) # [(speaker1, start_id_1), (speaker2, start_id_2)]
        split_indexes_2 = find_speaker_2_start_index(text2) # (speaker_2, start_index)

        # Compare data1's speaker 2 summary and data2's speaker 2 summary and select the shorter one
        if len(text1[split_indexes_1[1][1]:]) > len(text2[split_indexes_2[1]:]):
            data1[i]["output"] = text1[:split_indexes_1[1][1]] + text2[split_indexes_2[1]:]
    
    return data1



def main(args):
    with open(args.path, 'r') as f:
        data = json.load(f)

    if args.ensemble_path_1:
        print("Ensemble 1 ...")
        data = ensemble_1(data, args.ensemble_path_1)
        print("Ensemble 1 Done")
    elif args.ensemble_path_2:
        print("Ensemble 2 ...")
        data = ensemble_2(data, args.ensemble_path_2)
        print("Ensemble 2 Done")
    else:
        data = postprocess(data)

    with open(args.output_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)