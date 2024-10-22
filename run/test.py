import argparse
import json
import tqdm
import gc

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data import CustomDataset
from peft import PeftModel

# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, default="MLP-KTLim/llama-3-Korean-Bllossom-8B", required=True,help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
g.add_argument("--adapter_checkpoint_path", default=None, type=str, help="model path where model saved")
g.add_argument("--is_test", type=str, default='yes', help="dev or test data")
g.add_argument("--only_speaker_2", type=str, default="no", help="only speaker 2 model inference")
# fmt: on


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            low_cpu_mem_usage=True,
            revision = 'ece1eea3e0f1653eebac5016bdae1fbd37894078'
    )

    if args.adapter_checkpoint_path:
        print(f"{args.adapter_checkpoint_path} is loaded ...")
        model = PeftModel.from_pretrained(model, args.adapter_checkpoint_path)
        model = model.merge_and_unload()
        model.to(dtype = torch.bfloat16)
    model.eval()

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision = 'ece1eea3e0f1653eebac5016bdae1fbd37894078')
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # test or dev
    if args.is_test == 'yes':
        data = "test"
        dev_mode = False
        print("Choose test data ...")
    else:
        data = "dev"
        dev_mode = True
        print("Choose dev data ...")

    # only_speaker_2
    if args.only_speaker_2=="yes":
        print("only speaker 2 model inference selected ...")
        only_speaker_2 = True
    else:
        only_speaker_2 = False

    dataset = CustomDataset(f"resource/data/일상대화요약_{data}.json", tokenizer, dev_mode, only_speaker_2)

    with open(f"resource/data/일상대화요약_{data}.json", "r") as f:
        result = json.load(f)

    for idx in tqdm.tqdm(range(len(dataset))):
        inp = dataset[idx]
        outputs = model.generate(
            inp.to(args.device).unsqueeze(0),
            max_new_tokens=1024,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            repetition_penalty=1.1,
        )

        if args.is_test == 'yes':
            result[idx]["output"] = tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)
        else:
            result[idx]["inference"] = tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))
    
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    exit(main(parser.parse_args()))
