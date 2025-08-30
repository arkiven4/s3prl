Absolutely! Here’s the **exact call chain with line numbers and code snippets** for the downstream process when using Mockingjay as upstream, from `run_downstream.py` to your downstream expert’s `forward`:

---

## 1. `run_downstream.py`
**File:** run_downstream.py  
**Typical line:**  
```python
runner = Runner(args, config, ...)
runner.train()
```
- This instantiates the `Runner` class and starts training.

---

## 2. `downstream/runner.py`
**File:** runner.py  
**Relevant lines:**  
- **Upstream loading:**  
  ```python
  # runner.py, typically around line 100+
  self.upstream = get_upstream(...)
  ```
- **Downstream expert loading:**  
  ```python
  self.downstream = DownstreamExpert(...)
  ```
- **Training loop:**  
  ```python
  for batch in dataloader:
      wavs, labels = batch
      features = self.upstream(wavs)  # Calls UpstreamExpert.forward
      loss = self.downstream.forward(split, features, labels, records, ...)
  ```

---

## 3. expert.py
**File:** expert.py  
**Class:** `UpstreamExpert`  
- **Initialization:**  
  ```python
  22: class UpstreamExpert(UpstreamBase):
  27:     def __init__(self, ckpt, options_config=None, **kwargs):
  ...
  46:         self.transformer = PretrainedTransformer(options, inp_dim=-1)
  ```
- **Forward method:**  
  ```python
  51:     def forward(self, wavs):
  52:         last_hidden_state, hidden_states = self.transformer(wavs)
  54:         return {
  55:             "last_hidden_state": last_hidden_state,
  56:             "hidden_states": hidden_states.unbind(dim=0),
  57:         }
  ```
  - This is called by the runner, returning a dict with hidden states.

---

## 4. expert.py
**File:** expert.py  
**Class:** `DownstreamExpert`  
- **Forward method:**  
  ```python
  54:     def forward(self, split, features, labels, records, **kwargs):
  ...
  62:         features = pad_sequence(features, batch_first=True)
  63:         predicted = self.model(features, att_mask)
  68:         return loss
  ```
  - Receives `features` (from upstream), processes, and returns loss.

---

## **Summary Table**

| Step | File | Line(s) | Code Snippet |
|------|------|---------|--------------|
| 1 | run_downstream.py | ~ | `runner = Runner(...); runner.train()` |
| 2 | downstream/runner.py | ~100+ | `features = self.upstream(wavs)`<br>`loss = self.downstream.forward(...)` |
| 3 | expert.py | 51-57 | `def forward(self, wavs): ... return {"last_hidden_state": ..., "hidden_states": ...}` |
| 4 | expert.py | 54-68 | `def forward(self, split, features, labels, records, ...): ... return loss` |

---

## **Visual Call Chain**

```
run_downstream.py
  └── downstream/runner.py: features = self.upstream(wavs)
        └── upstream/mockingjay/expert.py:UpstreamExpert.forward (lines 51-57)
              └── returns {"last_hidden_state", "hidden_states"}
  └── downstream/runner.py: loss = self.downstream.forward(...)
        └── downstream/myown_souclass/expert.py:DownstreamExpert.forward (lines 54-68)
```

---

Let me know if you want the same for TERA or for any other step!