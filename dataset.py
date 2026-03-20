import random
import torch
from torch.utils.data import Dataset

# 1.)Tokenizer — basically just maps chars to integers so the model can read them.
class LogicTokenizer:
    def __init__(self):
        self.special = ["<PAD>", "<SOS>", "<EOS>"]
        self.digits = list("0123456789-")
        self.ops = list("+-*/=")
        
        self.full_vocab = self.special + self.digits + self.ops
        self.t2i = {char: i for i, char in enumerate(self.full_vocab)}
        self.i2t = {i: char for char, i in self.t2i.items()}
        
        # I am caching these ids here so the training loop stays clean.
        self.pad_id = self.t2i["<PAD>"]
        self.sos_id = self.t2i["<SOS>"]
        self.eos_id = self.t2i["<EOS>"]

    def encode(self, text):
        return [self.t2i[c] for c in text]

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # Filtering out the junk tokens so the output looks like actual math.
        return "".join(self.i2t[i] for i in ids if i not in {self.pad_id, self.sos_id, self.eos_id})

    def __len__(self):
        return len(self.full_vocab)

# 2.)Dataset class — this generates the math problems on the fly.
class LogicSynthDataset(Dataset):
    def __init__(self, n_examples=100000, min_d=1, max_d=8, math_ops="+-*/", seq_len=64, tokenizer=None):
        self.n_examples = n_examples
        self.min_d = min_d
        self.max_d = max_d
        self.math_ops = math_ops
        self.seq_len = seq_len
        self.tokenizer = tokenizer or LogicTokenizer()

    def __len__(self):
        return self.n_examples

    def _get_rand_num(self, digits):
        # Just a helper to make sure I get a number with the exact digit count I want.
        if digits == 1:
            return random.randint(0, 9)
        return random.randint(10**(digits-1), 10**digits - 1)

    def _generate_scratchpad(self, n1, n2, op):
        # This helps the model learn the carry-over logic by showing the step-by-step process.
        if op == '+':
            # Example: 12+39 -> "2+9=11,c1. 1+3+1=5."
            s1, s2 = str(n1)[::-1], str(n2)[::-1]
            max_l = max(len(s1), len(s2))
            steps = []
            carry = 0
            for i in range(max_l):
                d1 = int(s1[i]) if i < len(s1) else 0
                d2 = int(s2[i]) if i < len(s2) else 0
                sum_d = d1 + d2 + carry
                steps.append(f"{d1}+{d2}+{carry}={sum_d}")
                carry = sum_d // 10
            return "(" + ",".join(steps) + ")"
        return ""
    
    def __getitem__(self, index):
        # 3.)First we pick a random operator and lengths for both numbers.
        op = random.choice(list(self.math_ops))
        len1 = random.randint(self.min_d, self.max_d)
        len2 = random.randint(self.min_d, self.max_d)
        
        num1 = self._get_rand_num(len1)
        num2 = self._get_rand_num(len2)

        # 4.)Handling the math logic - Allowing negative results for subtraction.
        if op == "+": ans = num1 + num2
        elif op == "-":
            ans = num1 - num2
        elif op == "*": ans = num1 * num2
        elif op == "/":
            # Making sure I don't divide by zero and that the answer is always a clean integer.
            num2 = max(num2, 1)
            num1 = num1 - (num1 % num2)
            if num1 == 0: num1 = num2 * random.randint(1, 9)
            ans = num1 // num2

        # 5.)REVERSING the answer string — this is the "kill move" for the carry-over logic.
        query = f"{num1}{op}{num2}="
        ans_str = str(ans)[::-1] 

        # 6.)Time to tokenize and pad them so they can be batched easily.
        input_tokens = [self.tokenizer.sos_id] + self.tokenizer.encode(query)
        target_tokens = [self.tokenizer.sos_id] + self.tokenizer.encode(ans_str) + [self.tokenizer.eos_id]

        def pad_it(toks):
            # Pre-allocating a fixed-size tensor filled with the PAD id.
            # This is much faster than list concatenation like [0] * (len).
            tensor = torch.full((self.seq_len,), self.tokenizer.pad_id, dtype=torch.long)
            
            # Convert the raw tokens to a temporary tensor for the slice.
            raw_toks = torch.tensor(toks[:self.seq_len], dtype=torch.long)
            
            # Inject the tokens into the pre-allocated tensor from the start.
            # This ensures every sample is exactly the same shape for the DataLoader.
            tensor[:len(raw_toks)] = raw_toks
            
            return tensor
        
        return {
            "src": pad_it(input_tokens),
            "tgt": pad_it(target_tokens)
        }

# 7.)Simple test to make sure the strings and tensors look right.
if __name__ == "__main__":
    ds = LogicSynthDataset(n_examples=5, max_d=7)
    for i in range(len(ds)):
        item = ds[i]
        print(f"Sample {i}: {ds.tokenizer.decode(item['src'])} -> {ds.tokenizer.decode(item['tgt'])}")