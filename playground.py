import torch
from model import ModelConfig, LogicSynthTransformer
from dataset import LogicTokenizer
from eval import greedy_decode, load_best_checkpoint

# 1.)Setup—Loading the brain we just trained.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = LogicTokenizer()
cfg = ModelConfig(vocab_size=len(tokenizer), max_seq_len=128)

model = LogicSynthTransformer(cfg).to(DEVICE)
model = load_best_checkpoint(model)

# 2.)Playground loop—Type 'quit' to exit.
print("\n" + "="*50)
print("   LogicSynth Playground — Type your math problem!")
print("   Example: 123456+789012")
print("="*50 + "\n")

while True:
    problem = input("   Enter Problem: ").strip().replace(" ", "")
    if problem.lower() in ["quit", "exit"]: break
    
    if "=" not in problem: problem += "="
    
    # 3.) Decodes the FULL sequence (Scratchpad + Answer).
    pred_full = greedy_decode(model, problem)
    
    # 4.) Extracts only the answer after "Ans:" and flip it back.
    if "Ans:" in pred_full:
        # Splits "(1+1+c0=2)Ans:2" -> ["(1+1+c0=2)", "2"]
        raw_answer = pred_full.split("Ans:")[-1]
        answer = raw_answer[::-1]
        
        thought = pred_full.split("Ans:")[0]
        print(f"   Model Thought: {thought}")
    else:
        # Fallback if the model misses the "Ans:" token
        answer = pred_full[::-1]
    
    print(f"   Model Output: {answer}\n")