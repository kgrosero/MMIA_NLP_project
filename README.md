# MMIA-NLP CLass Project

## Build a Mini-Llama, a minimalist version of Llama2.

## Overview

In this assignment, you will implement important components of the Llama2 model to better understand its architecture. You will then perform sentence classification on the SST and CFIMDB datasets with this model.

## Assignment Details

### Your Tasks

You are responsible for implementing core components of Llama2 in the following files:
- `llama.py` - Main model architecture
- `classifier.py` - Classification head
- `optimizer.py` - AdamW optimizer  
- `rope.py` - Rotary position embeddings
- `lora.py` - LoRA implementation

You will work with `stories42M.pt`, an 8-layer, 42M parameter language model pretrained on the [TinyStories](https://arxiv.org/abs/2305.07759) dataset (machine-generated children's stories). This model is small enough to train with a small GPU.

### Background Theory

1. **LayerNorm (Layer Normalization)**
In this assignment, we will use LayerNorm instead of RMSNorm (which is used in Llama2). You will implement LayerNorm in llama.py. This normalization operates across features for each data point, reducing internal covariate shift, stabilizing gradients, and accelerating training.

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mi>a</mi>
        <mo stretchy="false">&#xAF;</mo>
      </mover>
    </mrow>
    <mi>i</mi>
  </msub>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <msub>
        <mi>a</mi>
        <mi>i</mi>
      </msub>
      <mo>&#x2212;</mo>
      <mi>&#x3BC;</mi>
    </mrow>
    <mi>&#x3C3;</mi>
  </mfrac>
  <msub>
    <mi>g</mi>
    <mi>i</mi>
  </msub>
  <mo>,</mo>
</math>

Here, āi is the normalized input, g is a learnable gain, and μ, σ are the mean and standard deviation of inputs a:

 <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>&#x3BC;</mi>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mi>n</mi>
  </mfrac>
  <munderover>
    <mo data-mjx-texclass="OP">&#x2211;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>n</mi>
    </mrow>
  </munderover>
  <msub>
    <mi>a</mi>
    <mi>i</mi>
  </msub>
  <mo>,</mo>
  <mstyle scriptlevel="0">
    <mspace width="1em"></mspace>
  </mstyle>
  <mi>&#x3C3;</mi>
  <mo>=</mo>
  <msqrt>
    <mfrac>
      <mn>1</mn>
      <mi>n</mi>
    </mfrac>
    <munderover>
      <mo data-mjx-texclass="OP">&#x2211;</mo>
      <mrow data-mjx-texclass="ORD">
        <mi>i</mi>
        <mo>=</mo>
        <mn>1</mn>
      </mrow>
      <mrow data-mjx-texclass="ORD">
        <mi>n</mi>
      </mrow>
    </munderover>
    <mo stretchy="false">(</mo>
    <msub>
      <mi>a</mi>
      <mi>i</mi>
    </msub>
    <mo>&#x2212;</mo>
    <mi>&#x3BC;</mi>
    <msup>
      <mo stretchy="false">)</mo>
      <mn>2</mn>
    </msup>
  </msqrt>
  <mo>.</mo>
</math>


<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>&#x3BC;</mi>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mi>n</mi>
  </mfrac>
  <munderover>
    <mo data-mjx-texclass="OP">&#x2211;</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>n</mi>
    </mrow>
  </munderover>
  <msub>
    <mi>a</mi>
    <mi>i</mi>
  </msub>
  <mo>,</mo>
  <mstyle scriptlevel="0">
    <mspace width="1em"></mspace>
  </mstyle>
  <mi>&#x3C3;</mi>
  <mo>=</mo>
  <msqrt>
    <mfrac>
      <mn>1</mn>
      <mi>n</mi>
    </mfrac>
    <munderover>
      <mo data-mjx-texclass="OP">&#x2211;</mo>
      <mrow data-mjx-texclass="ORD">
        <mi>i</mi>
        <mo>=</mo>
        <mn>1</mn>
      </mrow>
      <mrow data-mjx-texclass="ORD">
        <mi>n</mi>
      </mrow>
    </munderover>
    <mo stretchy="false">(</mo>
    <msub>
      <mi>a</mi>
      <mi>i</mi>
    </msub>
    <mo>&#x2212;</mo>
    <mi>&#x3BC;</mi>
    <msup>
      <mo stretchy="false">)</mo>
      <mn>2</mn>
    </msup>
  </msqrt>
  <mo>.</mo>
</math>
 
LayerNorm makes activations independent of input scale and weights to improve stability.

Reference: Layer Normalization (Ba et al., 2016)

### Testing Your Implementation

Once you have implemented the components, you will test your model in four settings:

1. **Text Generation**: Generate completions starting with: *"I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"*. You should see coherent, grammatical English (though content may be absurd due to the children's stories training data).

2. **Zero-shot Prompting**: Perform prompt-based sentiment analysis on SST-5 and CFIMDB datasets. This will give poor results (roughly random performance).

3. **Fine-tuning**: Perform task-specific fine-tuning with a classification head. This will give much stronger results.
4. **LoRA Fine-tuning**: Perform task-specific LoRA fine-tuning with a classification head. 

5. **Advanced Implementation (A+ requirement)**: Implement something new on top of the base requirements for potential extra credit.

### Important Notes

- Follow `setup.sh` to properly set up the environment and install dependencies
- See [structure.md](./structure.md) for detailed code structure descriptions
- Use only libraries installed by `setup.sh` - no external libraries (e.g., `transformers`) allowed
- The `data/cfimdb-test.txt` file contains placeholder labels (-1), so test accuracies may appear low
- Ensure reproducibility using the provided commands
- Do not change existing command options or add new required parameters
- Refer to [checklist.md](./checklist.md) for assignment requirements

## Reference Commands and Expected Results

### Text Generation
```bash
python run_llama.py --option generate
```
You should see continuations of the sentence `I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is...`. We will generate two continuations - one with temperature 0.0 (which should have a reasonably coherent, if unusual, completion) and one with temperature 1.0 (which introduces randomness for more creative and diverse outputs, though potentially less coherent).


### Zero-Shot Prompting

**SST Dataset:**
```bash
python run_llama.py --option prompt --batch_size 10 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt [--use_gpu]
```
- Dev Accuracy: 0.237 (0.000)
- Test Accuracy: 0.250 (0.000)

**CFIMDB Dataset:**
```bash
python run_llama.py --option prompt --batch_size 10 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt [--use_gpu]
```
- Dev Accuracy: 0.490 (0.000)
- Test Accuracy: 0.109 (0.000)

### Classification Fine-tuning

**SST Dataset:**
```bash
python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt [--use_gpu]
```
- Dev Accuracy: 0.411 (0.025)
- Test Accuracy: 0.399 (0.023)

**CFIMDB Dataset:**
```bash
python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt [--use_gpu]
```
- Dev Accuracy: 0.833 (0.060)
- Test Accuracy: 0.473 (0.198)

### LoRA Fine-tuning

**SST Dataset:**
```bash
python run_llama.py --option lora --epochs 5 --lr 2e-5 --batch_size 80 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-lora-output.txt --test_out sst-test-lora-output.txt --lora_rank 4 --lora_alpha 1.0 [--use_gpu]
```
- Dev Accuracy: 0.275 (0.024)
- Test Accuracy: 0.269 (0.020)

**CFIMDB Dataset:**
```bash
python run_llama.py --option lora --epochs 5 --lr 2e-5 --batch_size 10 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-lora-output.txt --test_out cfimdb-test-lora-output.txt --lora_rank 4 --lora_alpha 1.0 [--use_gpu]
```
- Dev Accuracy: 0.510 (0.051)
- Test Accuracy: 0.506 (0.239)

*Note: Mean reference accuracies over 10 random seeds with standard deviations in brackets.*

### Submission

The submission file should be a zip file with the following structure

```
SURNAME_FIRSTNAME/
├── run_llama.py
├── base_llama.py
├── llama.py
├── rope.py
├── lora.py
├── classifier.py
├── config.py
├── optimizer.py
├── sanity_check.py
├── tokenizer.py
├── utils.py
├── README.md
├── structure.md
├── checklist.md
├── sanity_check.data
├── generated-sentence-temp-0.txt
├── generated-sentence-temp-1.txt
├── sst-dev-prompting-output.txt
├── sst-test-prompting-output.txt
├── sst-dev-finetuning-output.txt
├── sst-test-finetuning-output.txt
├── sst-dev-lora-output.txt
├── sst-test-lora-output.txt
├── cfimdb-dev-prompting-output.txt
├── cfimdb-test-prompting-output.txt
├── cfimdb-dev-finetuning-output.txt
├── cfimdb-test-finetuning-output.txt
├── cfimdb-dev-lora-output.txt
├── cfimdb-test-lora-output.txt
└── setup.sh
```

### Acknowledgement

This code is based on llama2.c by Andrej Karpathy. Parts of the code are also from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)) and ANLP class at CMU.
