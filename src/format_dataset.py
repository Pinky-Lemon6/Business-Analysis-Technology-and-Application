# *************************************************************************
# Alpaca Format:
[
  {
    "instruction": "human instruction (required)",
    "input": "human input (optional)",
    "output": "model response (required)",
  }
]
# *************************************************************************



import os
import json
current_directory = os.path.dirname(os.path.abspath(__file__))

def main():
    dataset = []
    raw_data_dir = os.path.join(current_directory, "../data/comments-rated.csv")
    with open(raw_data_dir, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            lines.append(line)
        for line in lines[1: ]:
            tmp = line.split(",")
            number = tmp[0]
            scam = tmp[1]
            msg_length = tmp[2]
            msg = ",".join(tmp[3: ])
            data_line = {}
            data_line["instruction"] = "This is a review related to the financial sector on instgram, I want you to discern whether this review has the potential for financial fraud, and I want your answer to contain only true or false, where true means it's a financial fraud and false means it's not"
            data_line["input"] = msg
            data_line["output"] = "true" if scam=="1" else "false"
            dataset.append(data_line)
    output_dir = os.path.join(current_directory, "../data/alpaca_format_dataset.json")
    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()