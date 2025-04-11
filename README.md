# LLM Inference Boilerplate Code
Quick start for new LLM-inference project


### Setup
```bash
pip install -e . 
pip install flash-attn --no-build-isolation

git clone https://github.com/QwenLM/Qwen2.5-Math.git # use the math data published in 
cp -r Qwen2.5-Math/evaluation/data/ . 
rm -rf Qwen2.5-Math
```


### 