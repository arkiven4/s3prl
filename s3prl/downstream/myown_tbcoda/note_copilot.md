arkiven4: is there any example to make multitask in downstreamm

GitHub Copilot: Yes, there are examples and patterns for multitask learning in the s3prl downstream framework, though there is no single "multitask" template. The framework is flexible, so you can implement multitask learning by customizing your downstream task's expert.py.

### How to implement multitask learning

**1. Model:**  
Your model should output predictions for each task. You can define a model with multiple heads or outputs in your `model.py`.

**2. Dataset:**  
Your dataset should provide the necessary labels for each task in its `__getitem__`.

**3. DownstreamExpert:**  
In your expert.py, you need to:
- Accept and process multiple labels in the `forward` method.
- Compute losses for each task and combine them (e.g., sum or weighted sum).
- Log metrics for each task in `log_records`.

---

### Example: Multitask DownstreamExpert

Below is a simplified example of how you might structure a multitask downstream expert:

````python
# Example: expert.py for multitask learning

class DownstreamExpert(nn.Module):
    def __init__(self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs):
        super().__init__()
        self.model = MultiTaskModel(upstream_dim, **downstream_expert['modelrc'])
        self.loss_task1 = nn.CrossEntropyLoss()
        self.loss_task2 = nn.BCEWithLogitsLoss()
        # ... other initializations

    def forward(self, split, features, labels_task1, labels_task2, records, **kwargs):
        features = pad_sequence(features, batch_first=True)
        out1, out2 = self.model(features)
        loss1 = self.loss_task1(out1, labels_task1)
        loss2 = self.loss_task2(out2, labels_task2)
        loss = loss1 + loss2  # or weighted sum
        # Logging
        records['loss_task1'].append(loss1.item())
        records['loss_task2'].append(loss2.item())
        return loss

    def log_records(self, split, records, logger, global_step, **kwargs):
        logger.add_scalar(f'{split}/loss_task1', np.mean(records['loss_task1']), global_step)
        logger.add_scalar(f'{split}/loss_task2', np.mean(records['loss_task2']), global_step)
        # ... other logging
````

---

### References

- The [downstream/example/expert.py](https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/example/expert.py) ([local link](./example/expert.py)) is a minimal template you can extend for multitask.
- The README.md "Add new downstream tasks" section describes the requirements for a downstream task, which apply to multitask setups as well.
- You can look at downstream/fluent_commands/expert.py or downstream/atis/expert.py for more complex single-task examples.

---

**Summary:**  
There is no out-of-the-box multitask template, but you can implement multitask learning by extending the `DownstreamExpert` class to handle multiple outputs, losses, and metrics. Use the example above as a starting point.

arkiven4: is there any ways, like if i add 10 multitask, i not add it 1 by 1

GitHub Copilot: Yes, you can generalize your multitask setup so you **don't have to add each task manually**. The key is to use **lists or dictionaries** for your model heads, losses, and labels, and loop over them in your `forward` and logging functions.

Here’s a pattern you can follow:

---

#### 1. **Model: Multiple Heads**
Design your model so it returns a list or dict of outputs, one per task.

#### 2. **Losses: Store in a List/Dict**
Store your loss functions in a list or dict, e.g.:
```python
self.losses = [nn.CrossEntropyLoss() for _ in range(num_tasks)]
```

#### 3. **Forward: Loop Over Tasks**
Assume your dataset returns a list of labels (one per task) for each sample, and your model returns a list of predictions:
````python
def forward(self, split, features, labels_list, records, **kwargs):
    device = features[0].device
    features = pad_sequence(features, batch_first=True).to(device)
    preds = self.model(features)  # preds: list of [B, ...] per task

    total_loss = 0
    for i, (pred, labels, loss_fn) in enumerate(zip(preds, labels_list, self.losses)):
        labels = torch.LongTensor(labels).to(device)
        loss = loss_fn(pred, labels)
        total_loss += loss
        # Logging
        records[f'loss_{i}'].append(loss.item())
        # Add your metric here, e.g. accuracy

    return total_loss
````

#### 4. **Logging: Loop Over Tasks**
````python
def log_records(self, split, records, logger, global_step, **kwargs):
    for i in range(num_tasks):
        avg_loss = np.mean(records[f'loss_{i}'])
        logger.add_scalar(f'{split}/loss_{i}', avg_loss, global_step)
````

---

### **Summary**
- Use lists/dicts for losses, predictions, and labels.
- Loop over tasks in `forward` and logging.
- This way, you can scale to any number of tasks by just setting `num_tasks` and preparing your model/dataset accordingly.

---

**Tip:**  
You must also adapt your dataset and model to return/accept lists of labels and predictions for each task.

Let me know if you want a more concrete code example!