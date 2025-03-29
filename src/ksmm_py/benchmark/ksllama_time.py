import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from ksmm_py.layer.kronecker_sparse.interface import KSLinear
import argparse
import torch.utils.benchmark as benchmark
import pandas as pd
from pathlib import Path

from ksmm_py.benchmark.utils import (
    parse_patterns,
    get_dtype,
    set_device_and_get_device_name,
)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--up-pattern1", nargs='+', type=int)
    parser.add_argument("--up-pattern2", nargs='+', type=int)
    parser.add_argument("--down-pattern1", nargs='+', type=int)
    parser.add_argument("--down-pattern2", nargs='+', type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=196)
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--algo", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--min-run-time", type=float, default=5)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--saving-dir", type=Path, default="./results/2_llama")
    return parser.parse_args()


def replace_linear_layers(module, up_pattern1, up_pattern2, down_pattern1, down_pattern2, algo, device):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if child.in_features == 4096 and child.out_features == 14336:
                patterns = [up_pattern2, up_pattern1]
            elif child.in_features == 14336 and child.out_features == 4096:
                patterns = [down_pattern2, down_pattern1]
            else:
                continue
            new_linear = KSLinear(patterns=patterns, bias=child.bias is not None, algo=algo,
                                  dtype=child.weight.dtype, bs_last=False, device=device)
            setattr(module, name, new_linear)
        else:
            replace_linear_layers(child, up_pattern1, up_pattern2, down_pattern1, down_pattern2, algo, device)


def save_results(args, m, device_name):
    results_df = pd.DataFrame(
        [
            {
                "device_name": device_name,
                "up-pattern1": args.up_pattern1,
                "up-pattern2": args.up_pattern2,
                "down-pattern1": args.down_pattern1,
                "down-pattern2": args.down_pattern2,
                "batch-size": args.batch_size,
                "seq-length": args.seq_length,
                "precision": args.precision,
                "device": args.device,
                "algo": args.algo,
                "min-run-time": args.min_run_time,
                "device-id": args.device_id,
                "mean": m.mean,
                "median": m.median,
                "iqr": m.iqr,
            }
        ]
    )

    if args.saving_dir is not None:
        saving_path = args.saving_dir / f"algo={args.algo}-precision={args.precision}-batch_size={args.batch_size}-seq_length={args.seq_length}-up_pattern_1={args.up_pattern1}-up_pattern_2={args.up_pattern2}-down_pattern1={args.down_pattern1}-down_pattern2={args.down_pattern2}.csv"
        saving_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(saving_path)


if __name__ == "__main__":
    args = get_arguments()
    model_id = "meta-llama/Llama-3.1-8B"

    # SET DEVICE
    device_name = set_device_and_get_device_name(args.device_id, args.device)

    # GET PRECISION
    dtype = get_dtype(args.precision)

    # Prepare model
    print("Preparing model...")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.to(dtype=dtype, device=args.device)
    if args.algo != "dense":
        replace_linear_layers(model, args.up_pattern1, args.up_pattern2, args.down_pattern1, args.down_pattern2,
                              args.algo, args.device)
    print(model)

    # Prepare input
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    proust = "Longtemps, je me suis couché de bonne heure. Parfois, à peine ma bougie éteinte, mes yeux se fermaient si vite que je n’avais pas le temps de me dire : « Je m’endors. » Et, une demi-heure après, la pensée qu’il était temps de chercher le sommeil m’éveillait ; je voulais poser le volume que je croyais avoir encore dans les mains et souffler ma lumière ; je n’avais pas cessé en dormant de faire des réflexions sur ce que je venais de lire, mais ces réflexions avaient pris un tour un peu particulier ; il me semblait que j’étais moi-même ce dont parlait l’ouvrage : une église, un quatuor, la rivalité de François Ier et de Charles Quint. Cette croyance survivait pendant quelques secondes à mon réveil ; elle ne choquait pas ma raison mais pesait comme des écailles sur mes yeux et les empêchait de se rendre compte que le bougeoir n’était plus allumé. Puis elle commençait à me devenir inintelligible, comme après la métempsycose les pensées d’une existence antérieure ; le sujet du livre se détachait de moi, j’étais libre de m’y appliquer ou non ; aussitôt je recouvrais la vue et j’étais bien étonné de trouver autour de moi une obscurité, douce et reposante pour mes yeux, mais peut-être plus encore pour mon esprit, à qui elle apparaissait comme une chose sans cause, incompréhensible, comme une chose vraiment obscure. Je me demandais quelle heure il pouvait être ; j’entendais le sifflement des trains qui, plus ou moins éloigné, comme le chant d’un oiseau dans une forêt, relevant les distances, me décrivait l’étendue de la campagne déserte où le voyageur se hâte vers la station prochaine ; et le petit chemin qu’il suit va être gravé dans son souvenir par l’excitation qu’il doit à des lieux nouveaux, à des actes inaccoutumés, à la causerie récente et aux adieux sous la lampe étrangère qui le suivent encore dans le silence de la nuit, à la douceur prochaine du retour."
    inputs = tokenizer(proust, return_tensors="pt")
    max_length = inputs["input_ids"].shape[1]
    assert args.seq_length <= max_length
    for key in inputs:
        inputs[key] = inputs[key][:, :args.seq_length]
    inputs = {k: v.repeat(args.batch_size, 1) for k, v in inputs.items()}
    print("Batch-size, sequence length of input:", inputs["input_ids"].shape)
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    # Test one inference
    with torch.no_grad():
        outputs = model(**inputs)
    print("Shape of output logits: ", outputs.logits.shape)


    # BENCHMARK
    def measure():
        t = benchmark.Timer(
            stmt=f"llama_forward_pass(model, inputs)",
            setup=f"from ksmm_py.benchmark.utils import llama_forward_pass",
            globals={"model": model, "inputs": inputs},
            num_threads=torch.get_num_threads(),
            label=model_id,
            sub_label=f"{args.algo}, {args.precision}",
            description=f"(bs, seq-length) = {inputs["input_ids"].shape}",
        )
        return t.blocked_autorange(min_run_time=args.min_run_time)


    # Measure time. When not enough time to measure, increase the min_run_time to 11.0 * max(m.mean, m.median)
    m = measure()
    if m.number_per_run <= 10:
        args.min_run_time = 11.0 * max(m.mean, m.median)
        m = measure()
    print(m)

    # SAVE RESULTS
    save_results(args, m, device_name)
