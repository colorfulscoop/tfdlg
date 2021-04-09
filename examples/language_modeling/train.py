import pytorch_lightning as pl
import torch
import transformers


class BlockDataset(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, generator, block_size, drop_last=True):
        super().__init__()
        self._block_size = block_size
        self._tokenizer = tokenizer
        self._generator = generator
        self._drop_last = drop_last

    @classmethod
    def from_texts(cls, tokenizer, texts, block_size):
        return cls(tokenizer=tokenizer, generator=lambda: texts, block_size=block_size)

    @classmethod
    def from_file(cls, tokenizer, filepath, block_size):
        return cls(tokenizer=tokenizer,
                   generator=lambda: (line.strip("\n") for line in open(filepath)),
                   block_size=block_size
                  )

    def __iter__(self):
        """
            Yields (List[int])
        """
        ids = []
        for text in self._generator():
            ids.extend(self._tokenizer.encode(text))
            while len(ids) >= self._block_size+1:
                yield {"input_ids": ids[:self._block_size], "labels": ids[1:self._block_size+1]}
                ids = ids[self._block_size:]
        if not self._drop_last:
            yield ids


def collate_fn(item):
    """Collate function for DataLoader
    Args:
        item (List[dict[str, List[int]]]): BlockDataset のイテレータが返す辞書のリスト
    Returns:
        (dict[str, torch.Tensor]):
    """
    keys = item[0].keys()
    dic = {
        key: torch.tensor([x[key] for x in item])
        for key in keys
    }
    return dic


def forward(model, pad_token_id, src, tgt):
    output = model(input_ids=src)
    logits = output.logits  # shape: (batch_size, input_len, vocab_size)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    loss = loss_fn(
        input=logits.view(-1, logits.shape[-1]),
        target=tgt.view(-1)
    )
    return loss


class PLModel(pl.LightningModule):
    def __init__(self, config, lr, num_warmup_steps, num_training_steps):
        super().__init__()
        self.model = transformers.GPT2LMHeadModel(config)
        self._config = config
        self._lr = lr
        self._num_warmup_steps = num_warmup_steps
        self._num_training_steps = num_training_steps

    def training_step(self, batch, batch_idx):
        src, tgt = batch["input_ids"], batch["labels"]
        loss = forward(self.model, self._config.pad_token_id, src, tgt)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#validation-loop
        """
        src, tgt = batch["input_ids"], batch["labels"]
        loss = forward(self.model, self._config.pad_token_id, src, tgt)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        val_loss = sum(validation_step_outputs) / len(validation_step_outputs)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        if self._num_training_steps:
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self._num_warmup_steps,
                num_training_steps=self._num_training_steps
            )
        else:
            scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self._num_warmup_steps,
            )
        optimizers = [optimizer]
        schedulers = [{"scheduler": scheduler, "interval": "step"}]
        return optimizers, schedulers


def main(
    tokenizer_model, save_model_dir, train_file, valid_file,
    seed=None,
    block_size=1024,
    # [Model config]
    # for small
    n_layer=12, n_head=12, n_embd=768,
    # for medium -> n_layer=24, n_head=16, n_embd=1024
    # for large  -> n_layer=36, n_head=20, n_embd=5120
    # for XL     -> n_layer=48, n_head=24, n_embd=6400
    # [DataLoader options]
    batch_size=2, prefetch_factor=10, num_workers=1, shuffle_buffer_size=1000,
    lr=1e-4, num_warmup_steps=0, num_training_steps=None,
    # optoins for trainer
    **train_options
):
    # Set seed
    if seed:
        pl.seed_everything(seed)
    # Load tokenzier
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_model)

    # Prepare model
    config = transformers.GPT2Config(
        vocab_size=len(tokenizer),
        tokenizer_class=tokenizer.__class__.__name__,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
        cls_token_id=tokenizer.cls_token_id,
        unk_token_id=tokenizer.unk_token_id,
        #
        n_layer=n_layer, n_head=n_head, n_embd=n_embd
    )
    print(config)

    # Load data
    train_dataset = BlockDataset.from_file(
        block_size=config.n_ctx,
        tokenizer=tokenizer,
        filepath=train_file,
    )
    valid_dataset = BlockDataset.from_file(
        block_size=config.n_ctx,
        tokenizer=tokenizer,
        filepath=valid_file,
    )
    shuffled_train_dataset = torch.utils.data.BufferedShuffleDataset(
        train_dataset,
        buffer_size=shuffle_buffer_size,
    )

    # Build DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=shuffled_train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
    )

    # Trainer
    print("Training options:", train_options)
    pl_model = PLModel(
        config=config,
        lr=lr,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Setup callbacks
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
    ]
    if "gpus" in train_options:
        callbacks.append(pl.callbacks.GPUStatsMonitor())

    # Trainer
    trainer = pl.Trainer(
        **train_options,
        deterministic=True if seed else False,
        callbacks=callbacks,
    )
    trainer.fit(model=pl_model, train_dataloader=train_loader, val_dataloaders=valid_loader)
    pl_model.model.save_pretrained(save_model_dir)
    tokenizer.save_pretrained(save_model_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
