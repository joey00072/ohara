from transformers import AutoTokenizer
import optax
import wandb
import sys
from tqdm import tqdm
from itertools import cycle
from datasets import load_dataset
import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as sharding
from ramen import *

# wandb.login()

so = open("data.log", "w", 10)
sys.stdout.echo = so
sys.stderr.echo = so


ds = load_dataset("JeanKaddour/minipile", split="train")
ds_val = load_dataset("JeanKaddour/minipile", split="validation")

tokenizer = AutoTokenizer.from_pretrained(
    "NeelNanda/gpt-neox-tokenizer-digits", use_fast=True
)
tokenizer.padding_side = "right"

PAD = tokenizer.pad_token_id

microbatch_size = 32
accum_batch_size = microbatch_size * 4
vocab_size = len(tokenizer)
x_d = 768
x_n_min = 512
x_n_max = 2048
m_n = 64
layers = 14
dtype = jnp.bfloat16


def tokenize_fn(x):
    return tokenizer(x["text"], max_length=x_n_max, truncation=True)


def filter_fn(x):
    return len(x["input_ids"]) >= x_n_min


toks = (
    ds.map(tokenize_fn, batched=True, remove_columns=["text"], num_proc=128)
    .shuffle(seed=42)
    .filter(filter_fn, num_proc=128)
)
toks_val = ds_val.map(
    tokenize_fn, batched=True, remove_columns=["text"], num_proc=128
).filter(filter_fn, num_proc=128)


@eqx.filter_jit
def loss_fn_val(model, batch):
    seq = jax.vmap(lambda x: jnp.concatenate([jnp.array([PAD]), x]))(batch[:, :-1])

    logits = jax.vmap(model)(seq)

    losses = optax.softmax_cross_entropy_with_integer_labels(logits, batch)
    mask = jnp.where(batch != PAD, 1, 0)
    losses = losses * mask
    return losses.sum() / mask.sum()


loss_fn = eqx.filter_value_and_grad(loss_fn_val)

model = RAMENModel(vocab_size, layers, x_d, m_n, key=random.PRNGKey(0))

B, T, C = (3, 5, x_d)
seq = jax.random.uniform(random.PRNGKey(0), shape=(B, T), minval=0, maxval=vocab_size)
logits = jax.vmap(model)(seq)
exit(0)
if dtype != jnp.float32:
    model = jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if eqx.is_array(x) else x, model
    )


toks_cycle = cycle(toks)


def get_microbatch():
    microbatch = [next(toks_cycle)["input_ids"] for i in range(microbatch_size)]
    microbatch = [jnp.array(x) for x in microbatch]
    max_len = max([x.shape[0] for x in microbatch])
    microbatch = [
        jnp.pad(x, (0, max_len - x.shape[0]), constant_values=PAD) for x in microbatch
    ]
    microbatch = jnp.stack(microbatch)
    return microbatch


num_microbatches = accum_batch_size // microbatch_size
num_epochs = 1
num_batches = num_epochs * len(toks) // accum_batch_size

schedule = optax.cosine_onecycle_schedule(num_batches, 2e-4)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.lion(schedule))
optim = optax.MultiSteps(optimizer, every_k_schedule=num_microbatches)

opt_state = optim.init(eqx.filter(model, eqx.is_array))


def Updater(optimizer):
    @eqx.filter_jit
    def train_step(model, opt_state, batch):
        loss, grads = loss_fn(model, batch)
        model_arrays = eqx.filter(model, eqx.is_array)
        updates, opt_state = optimizer.update(grads, opt_state, model_arrays)
        model_arrays = eqx.apply_updates(model_arrays, updates)

        model = eqx.combine(model_arrays, model)

        return loss, model, opt_state

    return train_step


params_count = sum(
    x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
)
print(f"{params_count / 1000000}M Parameters")

train_step = Updater(optim)


def validate(model, toks):
    acum_loss = 0.0

    idx = 0
    count = len(toks_val) // microbatch_size

    for i in range(count):
        microbatch = [toks_val[idx + i]["input_ids"] for i in range(microbatch_size)]
        microbatch = [jnp.array(x) for x in microbatch]
        max_len = max([x.shape[0] for x in microbatch])
        microbatch = [
            jnp.pad(x, (0, max_len - x.shape[0]), constant_values=PAD)
            for x in microbatch
        ]
        microbatch = jnp.stack(microbatch)
        microbatch = jax.device_put(microbatch, shard)
        idx += microbatch_size

        acum_loss += loss_fn_val(model, microbatch)

    acum_loss /= count
    return acum_loss


num_devices = len(jax.devices())
devices = mesh_utils.create_device_mesh((num_devices, 1))
shard = sharding.PositionalSharding(devices)


# run = wandb.init(project='memnet-small-1epoch', entity='jpritsk')

for epoch in range(num_epochs):
    for batch in tqdm(range(num_batches)):
        accum_loss = 0.0

        for _i in range(num_microbatches):
            microbatch = get_microbatch()
            microbatch = jax.device_put(microbatch, shard)
            loss, model, opt_state = train_step(model, opt_state, microbatch)
            accum_loss += loss
            assert not jnp.isnan(accum_loss.item())

        accum_loss /= num_microbatches

        if (batch + 1) % 20 == 0:
            wandb.log({"train_loss": accum_loss.item()}, step=batch + 1)

        if (batch + 1) % 100 == 0 or batch == num_batches - 1:
            val_loss = validate(model, toks_val)
            # wandb.log({'val_loss': val_loss.item()}, step=batch + 1)
            print(f"\nStep: {batch + 1}/{num_batches}, loss: {val_loss.item()}")

    eqx.tree_serialise_leaves(f"model{epoch}.eqx", model)
    wandb.save(self_path)
    wandb.save(f"model{epoch}.eqx")

# wandb.finish()
