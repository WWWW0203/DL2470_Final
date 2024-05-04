import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from datasets import Dataset, load_dataset
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from pytorch_lightning.utilities.data import DataLoader
import torch
import random
import os
import copy

IGNORE_INDEX = -100
TITLE = "The Chronicles of Eldoria: The Forgotten Kingdom"
SCENARIO = "In the mystical land of Eldoria, where magic intertwines with the threads of reality, you find yourself waking up in a small, dimly lit hut. The scent of damp wood and old parchment fills the air. You are an apprentice to the old sage, Eledor, who has mysteriously disappeared. The only clue to his whereabouts is a cryptic note left on his desk, reading, \"Seek the Forgotten Kingdom, where shadows whisper secrets.\"\n\nYou step outside the hut, and the village of Elden greets you. The cobblestone paths, the thatched-roof houses, and the distant sound of the blacksmith's hammer hitting the anvil are all too familiar. Yet, an air of unease hangs over the village. The villagers, once cheerful and bustling with life, now wear worried expressions. Eledor's disappearance has not gone unnoticed.\n\nTo the north lies the dense Eldwood Forest, a place of enchantment and danger, where mythical creatures lurk in the shadows. To the east, the towering Eldoria Mountains stand guard, their peaks lost in the clouds. To the west, the tranquil Elden Lake mirrors the sky, hiding ancient secrets beneath its surface. And to the south, the vast Eldoria Plains stretch out, leading to unknown lands.\n\nYour adventure begins here, in the heart of Eldoria. Will you brave the Eldwood Forest, scale the Eldoria Mountains, dive into the depths of Elden Lake, or traverse the Eldoria Plains? The choice is yours. Remember, the fate of Eledor and the entire village of Elden rests on your shoulders. Seek the Forgotten Kingdom, decipher its secrets, and save your mentor. The Chronicles of Eldoria await you."


class StatusDataModule(pl.LightningDataModule):
    
    def __init__(self, args, tokenizer) -> None:
        super(StatusDataModule, self).__init__()
        
        self.dataset = Dataset.load_from_disk(os.path.join(args.input_dir, "result_status"))
        self.seed = args.seed
        
        if args.shuffle:
            self.dataset = self.dataset.shuffle(seed=self.seed)

        self.batch_size = args.batch_size
        self.tokenizer = tokenizer    

    def collate_fn(self, batch):
        batch = [b.values() for b in batch]
        results, status = list(zip(*batch))        
        batch = self.tokenizer(results, return_tensors="pt", padding="longest", max_length=512, truncation=True)
        labels = []
        for s in status:
            if s == "Win":
                labels.append(0)
            elif s == "Dead":
                labels.append(1)
            elif s == "Wounded":
                labels.append(2)
            elif s == "Healed":
                labels.append(3)
            elif s == "Nothing happened":
                labels.append(4)
            else:
                raise ValueError(f"Invalid status: {s}")
        
        labels = torch.LongTensor(labels)
        batch["labels"] = labels
        return batch

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
    
    
    

class ResultDataModule(pl.LightningDataModule):
    
    def __init__(self, args, tokenizer) -> None:
        super(ResultDataModule, self).__init__()
        
        self.dataset = Dataset.load_from_disk(os.path.join(args.input_dir, "action_outcome"))
        self.seed = args.seed
        
        if args.shuffle:
            self.dataset = self.dataset.shuffle(seed=self.seed)

        self.batch_size = args.batch_size
        self.tokenizer = tokenizer
        
    def _tokenize_fn(self, strings):
        """Tokenize a list of strings."""
        tokenized_list = [
            self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=1024,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(
        self,
        sources,
        targets,
    ):
        """Preprocess the data by tokenizing."""
        targets = [target + ' ' + self.tokenizer.eos_token for target in targets]
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [self._tokenize_fn(strings) for strings in (examples, sources)]    
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
                
        # input_ids = [ids[:-1] for ids in input_ids] # remove the last token
        # labels = [label[1:] for label in labels]    # remove the first token
        
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len - 1] = IGNORE_INDEX
            
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        if len(sources) == len(targets) == 1: 
            return dict(input_ids=input_ids[0], labels=labels[0], attention_mask=input_ids[0].ne(self.tokenizer.pad_token_id))
        else:
            return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
    

    def collate_fn(self, batch):
        batch = [b.values() for b in batch]
        cur_situations, actions, results, new_situations = list(zip(*batch))
        
        source_texts = [f"Title: {TITLE}\n\nBackground: {SCENARIO}\n\nCurrent Situation: {cur_situation}\n\nAction: {action}\n\n" for cur_situation, action in zip(cur_situations, actions)]
        target_texts = [f"Result: {result}\n\nNew Situation: {new_situation}\n\n" for result, new_situation in zip(results, new_situations)]
        
        batch = self.preprocess(sources=source_texts, targets=target_texts)
        return batch

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
    

class ActionDataModule(pl.LightningDataModule):
    
    def __init__(self, args, tokenizer) -> None:
        super(ActionDataModule, self).__init__()
        
        self.dataset = Dataset.load_from_disk(os.path.join(args.input_dir, "outcome_action"))
        self.seed = args.seed
        
        if args.shuffle:
            self.dataset = self.dataset.shuffle(seed=self.seed)
        
        self.batch_size = args.batch_size
        self.tokenizer = tokenizer
        
    def _tokenize_fn(self, strings):
        """Tokenize a list of strings."""
        tokenized_list = [
            self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=1024,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(
        self,
        sources,
        targets,
    ):
        """Preprocess the data by tokenizing."""
        targets = [target + ' ' + self.tokenizer.eos_token for target in targets]
        # targets = targets + ' ' + self.tokenizer.eos_token
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [self._tokenize_fn(strings) for strings in (examples, sources)]    
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        
        # input_ids = [ids[:-1] for ids in input_ids] # remove the last token
        # labels = [label[1:] for label in labels]    # remove the first token
        
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len - 1] = IGNORE_INDEX
            
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        if len(sources) == len(targets) == 1: 
            return dict(input_ids=input_ids[0], labels=labels[0], attention_mask=input_ids[0].ne(self.tokenizer.pad_token_id))
        else:
            return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
    

    def collate_fn(self, batch):
        batch = [b.values() for b in batch]
        cur_situations, actions = list(zip(*batch))
        
        source_texts = [f"Title: {TITLE}\n\nBackground: {SCENARIO}\n\nCurrent Situation: {cur_situation}\n\n" for cur_situation in cur_situations]
        target_texts = [f"Action: {action}" for action in actions]
        
        batch = self.preprocess(sources=source_texts, targets=target_texts)
        return batch

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)


class Model(pl.LightningModule):
    
    def __init__(self, model, tokenizer, lr, seed) -> None:
        super(Model, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.seed = seed
        self.lr = lr

    def save_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        print("Saving model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def on_train_start(self) -> None:
        seed_everything(self.seed)
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


