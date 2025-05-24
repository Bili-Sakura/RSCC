import json


def convert_to_qwenvl(input_path, output_file):
    with open(input_path, "r") as infile:
        for line in infile:
            data = json.loads(line)

            # Create Qwen-VL format entry
            qwen_entry = {
                "images": [data["pre_image"], data["post_image"]],
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\n<image>\nDescribe the changes between these two satellite images in a news style with a few sentences.",
                    },
                    {"from": "gpt", "value": data["change_caption"]},
                ],
            }

            # Write to output file
            output_file.write(json.dumps(qwen_entry) + "\n")


if __name__ == "__main__":
    output_jsonl = "path/to/rscc_subset_qwenvl_format.jsonl"

    with open(output_jsonl, "w") as outfile:
        # Process XBD dataset
        convert_to_qwenvl("path/to/xbd_gt_qwen25vl72b.jsonl", outfile)

        # Process EBD dataset
        convert_to_qwenvl("path/to/ebd_gt_qwen25vl72b.jsonl", outfile)

        convert_to_qwenvl("path/to/xbd_subset_qvq.jsonl", outfile)
